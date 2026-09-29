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

"""DAWN-crop mechanism package: the load-bearing controls for the win-win band,
re-measured on the detection-dataset testbed (replaces the CIFAR versions).

  (1) PTQ ABLATION (decisive): same FP32 model, W8A8 calibrate-only (no QAT),
      evaluated on the same cells -> the band must NOT appear (flat negative).
  (2) EXTENDED SWEEP: gaussian sigma {10,12,14,16} for FP32/QAT -> band decay /
      reversal location.
  (3) SPECTRAL SURROGATE: per-rank energy profiles (FP32 clean-test SVD basis) of
      the quantization residual vs the blur residual -> profile cosine.
  (4) MARGIN PROBE: W8A8-FP32 logit-margin gap at clean / sigma4 / sigma8.
  (5) EFFECTIVE RANK: FP32 vs W8A8 feature effective rank (unchanged expected).

Same data pipeline and equalized-head protocol as dino_dawncrops_winwin.py.
PTQ cells over 3 seeds; probes (3)-(5) on seed 0, matching the CIFAR protocol.
Env: QAT_STEPS=400  HEAD_STEPS=3000  SEEDS=3
"""
import os
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
import numpy as np
np.float_ = np.float64; np.unicode_ = np.str_
import warnings; warnings.filterwarnings("ignore")
import copy, json
import torch
import torch.nn as nn

from dino_rotlsq_engine import (quantize_backbone, calibrate, qat_train, DinoClassifier,
                                DEV, BATCH)
from dino_crash_winwin import loader as tensor_loader
from dino_dawncrops_winwin import (build_crops, split_by_image, tensors, make_backbone,
                                   extract_feats, train_head_on_feats, eval_model,
                                   CLASSES, QAT_STEPS, HEAD_STEPS, SEEDS)

BASE_SIGMAS = [0, 1, 2, 3, 4, 6, 8]
EXT_SIGMAS = [10, 12, 14, 16]


@torch.inference_mode()
def feats_logits(model, xt):
    model.eval(); F, L = [], []
    for b in range(0, len(xt), BATCH):
        x = xt[b:b+BATCH].to(DEV)
        f = model.backbone(x)
        F.append(f.cpu()); L.append(model.head(f).cpu())
    return torch.cat(F).numpy(), torch.cat(L)


def margin_gap(Lq, Lf, y):
    def margin(L):
        yt = torch.as_tensor(y)
        true = L[torch.arange(len(yt)), yt]
        L2 = L.clone(); L2[torch.arange(len(yt)), yt] = -1e9
        return (true - L2.max(1).values).mean().item()
    return margin(Lq) - margin(Lf)


def eff_rank(F):
    s = np.linalg.svd(F - F.mean(0), compute_uv=False)
    p = s**2 / (s**2).sum()
    return float(np.exp(-(p * np.log(p + 1e-12)).sum()))


def main():
    print("=== DAWN-crop mechanism package ===", flush=True)
    x, y, img_ids = build_crops()
    tr_idx, te_idx = split_by_image(y, img_ids)
    tr_xt = tensors(x[tr_idx]); tr_y = y[tr_idx]
    te_y = y[te_idx]

    cells_xt = {f"gauss_s{s:g}" if s else "clean": tensors(x[te_idx], sigma=float(s) if s else None)
                for s in BASE_SIGMAS + EXT_SIGMAS}
    for fam in ("defocus_blur", "motion_blur"):
        for v in (4, 5):
            cells_xt[f"{fam}_sev{v}"] = tensors(x[te_idx], corruption=fam, sev=v)
    print("cells materialized", flush=True)

    fp_backbone = make_backbone().to(DEV).eval()
    tr_feats = extract_feats(fp_backbone, tr_xt)

    out = {"ptq": {}, "ext_sweep": {}, "probes": {}}
    probe_models = None
    for s in range(SEEDS):
        torch.manual_seed(s); np.random.seed(s)
        head_c = train_head_on_feats(nn.Linear(384, len(CLASSES)), tr_feats, tr_y,
                                     HEAD_STEPS, batch=256)
        fp_head = train_head_on_feats(copy.deepcopy(head_c), tr_feats, tr_y,
                                      QAT_STEPS, batch=BATCH)
        fp = DinoClassifier(fp_backbone, fp_head).to(DEV).eval()
        # QAT model (band) -- same as winwin run
        qm = DinoClassifier(quantize_backbone(make_backbone()).to(DEV),
                            copy.deepcopy(head_c).to(DEV)).to(DEV)
        trl = tensor_loader(tr_xt, tr_y, True)
        calibrate(qm, trl); qat_train(qm, trl, QAT_STEPS)
        # PTQ ablation: calibrate-only, NO QAT; head = the FP32 head (same budget,
        # no co-adaptation) -> isolates INT8 arithmetic from trained invariance
        pq = DinoClassifier(quantize_backbone(make_backbone()).to(DEV),
                            copy.deepcopy(fp_head).to(DEV)).to(DEV)
        calibrate(pq, trl)
        print(f"[seed {s}] trained (fp/qat/ptq)", flush=True)

        for name, xt in cells_xt.items():
            rf = eval_model(fp, xt, te_y)
            rq = eval_model(qm, xt, te_y)
            rp = eval_model(pq, xt, te_y)
            out["ptq"].setdefault(name, []).append(
                dict(fp32=rf["bal_acc"], qat=rq["bal_acc"], ptq=rp["bal_acc"]))
            print(f"[{name:>16} seed {s}] fp {rf['bal_acc']:.2f} qat {rq['bal_acc']:.2f} "
                  f"({rq['bal_acc']-rf['bal_acc']:+.2f}) ptq {rp['bal_acc']:.2f} "
                  f"({rp['bal_acc']-rf['bal_acc']:+.2f})", flush=True)
        if s == 0:
            probe_models = (fp, qm)

    # ---- probes on seed 0 ----
    fp, qm = probe_models
    F_fp_clean, L_fp_clean = feats_logits(fp, cells_xt["clean"])
    F_q_clean, L_q_clean = feats_logits(qm, cells_xt["clean"])
    F_fp_s6, _ = feats_logits(fp, cells_xt["gauss_s6"])

    Xc = F_fp_clean - F_fp_clean.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    quant_res = (F_q_clean - F_fp_clean) @ Vt.T
    blur_res = (F_fp_s6 - F_fp_clean) @ Vt.T
    eq = (quant_res**2).mean(0); eb = (blur_res**2).mean(0)
    cos = float(eq @ eb / (np.linalg.norm(eq) * np.linalg.norm(eb)))
    eqc, ebc = eq - eq.mean(), eb - eb.mean()
    corr = float(eqc @ ebc / (np.linalg.norm(eqc) * np.linalg.norm(ebc)))
    out["probes"]["profile_cosine"] = cos
    out["probes"]["profile_corr_meanremoved"] = corr
    out["probes"]["eff_rank_fp32"] = eff_rank(F_fp_clean)
    out["probes"]["eff_rank_w8a8"] = eff_rank(F_q_clean)

    mg = {}
    for cell in ("clean", "gauss_s4", "gauss_s8", "gauss_s14"):
        _, Lf = feats_logits(fp, cells_xt[cell])
        _, Lq = feats_logits(qm, cells_xt[cell])
        mg[cell] = margin_gap(Lq, Lf, te_y)
    out["probes"]["margin_gap"] = mg

    print("\nPROBES:", json.dumps(out["probes"], indent=1), flush=True)
    json.dump(out, open("dino_dawncrops_mechanism.json", "w"), indent=2)
    print("\nDONE -> dino_dawncrops_mechanism.json", flush=True)


if __name__ == "__main__":
    main()
