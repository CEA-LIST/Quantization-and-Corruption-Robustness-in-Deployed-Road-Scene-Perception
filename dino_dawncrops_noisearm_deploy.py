# Copyright (c) 2026 CEA (Commissariat à l'énergie atomique et aux énergies alternatives)
# Authors: Aymen Bouguerra, Ansgar Radermacher, Fabio Arnez, Chokri Mraidha
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The two decisive reviewer experiments, run on the EXACT dino_dawncrops_winwin
protocol (same crops, same by-image split, same equalized budgets, same cells):

ARM 1 (fp32)     : head_c + 400 no-quant continuation steps (the paper's FP32 arm).
ARM 2 (w8a8 sim) : head_c + calibrate + 400 QAT steps, fake-quant (the paper's arm).
ARM 3 (noise)    : head_c + 400 steps with MATCHED-MAGNITUDE injected noise at the
                   exact 48 quantizer insertion points: input activations get
                   U(-D/2, +D/2) with D = the calibrated (pre-QAT) per-tensor
                   activation scale of the SAME model; weights get per-channel
                   U(-Dw/2, +Dw/2) with Dw = amax/127 (the weight-quantizer scale),
                   resampled every forward. No grid, no clipping: pure noise
                   regularization of quantization magnitude. Backbone frozen, head
                   trained (same optimizer/lr/steps as the QAT arm's head group).
                   EVALUATED WITH NOISE OFF (a plain FP32 deploy).
(The paper's ARM 4, the QAT arm deployed on true-INT8 kernels, is not part of this
release; its measured numbers are in results/dino_dawncrops_noisearm_deploy.json.)

PRE-REGISTERED READINGS (before running):
  R1  If ARM 3 reproduces the Gaussian severity 4-6 balanced-accuracy band
      (+2.5/+4.1 at sigma 4/6), the effect is generic noise regularization.
      If ARM 3 stays flat/negative there, the effect is quantization-specific
      (the grid/clipping/LSQ adaptation, not the noise magnitude alone).
Env: QAT_STEPS=400 HEAD_STEPS=3000 SEEDS=3 (paper protocol); SMOKE=1 for dry run.
"""
import os
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
import numpy as np
np.float_ = np.float64; np.unicode_ = np.str_
import warnings; warnings.filterwarnings("ignore")
import copy, json
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from dino_rotlsq_engine import (quantize_backbone, calibrate, qat_train, DinoClassifier,
                                DEV, BATCH)
from quantization.modules import QuantizedLinearLayer
from dino_crash_winwin import loader as tensor_loader
from dino_dawncrops_winwin import (build_crops, split_by_image, tensors, make_backbone,
                                   extract_feats, train_head_on_feats, eval_model,
                                   CLASSES, CELLS)

SMOKE = os.environ.get("SMOKE", "0") == "1"
QAT_STEPS = int(os.environ.get("QAT_STEPS", "20" if SMOKE else "400"))
HEAD_STEPS = int(os.environ.get("HEAD_STEPS", "100" if SMOKE else "3000"))
SEEDS = int(os.environ.get("SEEDS", "1" if SMOKE else "3"))
if SMOKE:
    CELLS = [c for c in CELLS if c[0] in ("clean", "gauss_s4", "gauss_s6")]
OUT_JSON = "dino_dawncrops_noisearm_deploy.json"


class MatchedNoiseLinear(nn.Module):
    """FP32 linear with quantization-magnitude uniform noise at the quantizer
    insertion points, active in train mode only. No grid, no clipping."""
    def __init__(self, lin, a_scale):
        super().__init__()
        self.lin = lin
        self.register_buffer("a_scale", torch.tensor(float(a_scale)))
        w = lin.weight.data
        self.register_buffer("w_scale",
                             (w.abs().amax(dim=1, keepdim=True) / 127.0).clamp(min=1e-8))

    def forward(self, x):
        if self.training:
            x = x + (torch.rand_like(x) - 0.5) * self.a_scale
            w = self.lin.weight + (torch.rand_like(self.lin.weight) - 0.5) * self.w_scale
            return Fn.linear(x, w, self.lin.bias)
        return self.lin(x)


def act_scales_of(qmodel):
    """Per-layer calibrated activation scales, keyed by QuantizedLinearLayer name."""
    return {name: float(mod.activation_quantizer.scale.data.float().reshape(()))
            for name, mod in qmodel.named_modules()
            if isinstance(mod, QuantizedLinearLayer)}


def build_noise_backbone(scales):
    bb = make_backbone()
    n = 0
    for name, mod in list(bb.named_modules()):
        if isinstance(mod, nn.Linear) and "blocks" in name:
            parent = bb
            for p in name.split(".")[:-1]:
                parent = getattr(parent, p) if not p.isdigit() else parent[int(p)]
            setattr(parent, name.split(".")[-1], MatchedNoiseLinear(mod, scales[name]))
            n += 1
    assert n == len(scales), (n, len(scales))
    return bb


def noise_train(model, loader_, steps):
    """Mirror of qat_train minus the LSQ scale group: head only, same lr, noise
    active via train mode."""
    model.train()
    for p in model.parameters():
        p.requires_grad = False
    for p in model.head.parameters():
        p.requires_grad = True
    opt = torch.optim.AdamW(model.head.parameters(), lr=1e-3)
    it = iter(loader_); done = 0
    while done < steps:
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(loader_); x, y = next(it)
        x, y = x.to(DEV), y.to(DEV)
        loss = Fn.cross_entropy(model(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
        done += 1
    return model.eval()


def main():
    print(f"=== noise-arm sweep (SMOKE={SMOKE}, seeds={SEEDS}, "
          f"qat={QAT_STEPS}, head={HEAD_STEPS}) ===", flush=True)
    x, y, img_ids = build_crops()
    tr_idx, te_idx = split_by_image(y, img_ids)
    import collections
    print(f"crops {len(y)} | train {len(tr_idx)} {collections.Counter(y[tr_idx].tolist())} "
          f"| test {len(te_idx)} {collections.Counter(y[te_idx].tolist())}", flush=True)

    tr_xt = tensors(x[tr_idx]); tr_y = y[tr_idx]
    cells_xt = {}
    for name, fam, sev, sigma in CELLS:
        cells_xt[name] = tensors(x[te_idx], corruption=fam, sev=sev, sigma=sigma)
    te_y = y[te_idx]
    print("eval cells materialized", flush=True)

    fp_backbone = make_backbone().to(DEV).eval()
    tr_feats = extract_feats(fp_backbone, tr_xt)

    ARMS = ("fp32", "w8a8", "noise")
    results = {name: {a: [] for a in ARMS} for name in cells_xt}
    noise_scale_stats = []

    for s in range(SEEDS):
        # --- identical op order to dino_dawncrops_winwin for fp32/w8a8 ---
        torch.manual_seed(s); np.random.seed(s)
        head_c = train_head_on_feats(nn.Linear(384, len(CLASSES)), tr_feats, tr_y,
                                     HEAD_STEPS, batch=256)
        fp_head = train_head_on_feats(copy.deepcopy(head_c), tr_feats, tr_y,
                                      QAT_STEPS, batch=BATCH)
        fp = DinoClassifier(fp_backbone, fp_head).to(DEV).eval()
        qm = DinoClassifier(quantize_backbone(make_backbone()).to(DEV),
                            copy.deepcopy(head_c).to(DEV)).to(DEV)
        trl = tensor_loader(tr_xt, tr_y, True)
        calibrate(qm, trl)
        scales = act_scales_of(qm.backbone)   # calibrated, pre-QAT: the matched noise
        qat_train(qm, trl, QAT_STEPS)
        print(f"[seed {s}] fp32 + w8a8 trained | act-scale mean "
              f"{np.mean(list(scales.values())):.4f} "
              f"range [{min(scales.values()):.4f}, {max(scales.values()):.4f}]", flush=True)
        noise_scale_stats.append(dict(mean=float(np.mean(list(scales.values()))),
                                      min=float(min(scales.values())),
                                      max=float(max(scales.values()))))

        # --- ARM 3: matched-noise fp32 ---
        torch.manual_seed(s); np.random.seed(s)
        nm = DinoClassifier(build_noise_backbone(scales).to(DEV),
                            copy.deepcopy(head_c).to(DEV)).to(DEV)
        noise_train(nm, tensor_loader(tr_xt, tr_y, True), QAT_STEPS)
        print(f"[seed {s}] noise arm trained", flush=True)


        for name, xt in cells_xt.items():
            rf = eval_model(fp, xt, te_y)
            rq = eval_model(qm, xt, te_y)
            rn = eval_model(nm, xt, te_y)          # eval mode -> noise OFF
            for a, r in zip(ARMS, (rf, rq, rn)):
                results[name][a].append(r)
            print(f"[{name:>16} seed {s}] bal fp {rf['bal_acc']:.2f} | "
                  f"w8a8 {rq['bal_acc']:.2f} ({rq['bal_acc']-rf['bal_acc']:+.2f}) | "
                  f"noise {rn['bal_acc']:.2f} ({rn['bal_acc']-rf['bal_acc']:+.2f})",
                  flush=True)
        del qm, nm, fp
        torch.cuda.empty_cache()

    summary = {}
    print(f"\n{'cell':>16}{'metric':>8}{'FP32':>13}{'W8A8':>13}{'NOISE':>13}"
          f"{'dQAT':>8}{'dNOISE':>8}")
    for name in results:
        summary[name] = {}
        for met in ("bal_acc", "acc", "ece", "aurc"):
            row = {}
            for a in ARMS:
                v = [r[met] for r in results[name][a]]
                row[a] = [float(np.mean(v)), float(np.std(v))]
            row["delta_qat"] = row["w8a8"][0] - row["fp32"][0]
            row["delta_noise"] = row["noise"][0] - row["fp32"][0]
            summary[name][met] = row
            if met == "bal_acc":
                print(f"{name:>16}{met:>8}"
                      f"{row['fp32'][0]:>9.2f}±{row['fp32'][1]:<3.1f}"
                      f"{row['w8a8'][0]:>9.2f}±{row['w8a8'][1]:<3.1f}"
                      f"{row['noise'][0]:>9.2f}±{row['noise'][1]:<3.1f}"
                      f"{row['delta_qat']:>+8.2f}{row['delta_noise']:>+8.2f}", flush=True)

    json.dump({"summary": summary, "per_seed": {n: {a: results[n][a] for a in ARMS}
                                                for n in results},
               "noise_scales": noise_scale_stats,
               "seeds": SEEDS, "qat_steps": QAT_STEPS, "head_steps": HEAD_STEPS,
               "smoke": SMOKE,
               "design": "noise arm: U(-D/2,D/2) at 48 quantizer sites, calibrated "
                         "pre-QAT act scales + amax/127 per-channel weight scales, "
                         "resampled per forward, train-only, eval noise-off"},
              open(OUT_JSON, "w"), indent=2)
    print(f"\nDONE -> {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
