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

"""DAWN-crop probe package 2: the remaining refutation probes on the
detection-dataset testbed.

  (1) INPUT-FREQUENCY SENSITIVITY (refutes "input low-pass"): sinusoidal
      gratings injected into clean test crops; W8A8/FP32 logit-shift ratio per
      spatial frequency. CIFAR result was ratio >= 1 rising at high f.
  (2) LOW-PASS SPECIFICITY (fingerprint control): high-frequency corruptions
      (gaussian noise, pixelate) at severities 3/5 -> the gain should NOT
      appear (neutral-to-negative), unlike the low-pass blur families.
  (3) CLASS-BALANCED MARGIN GAP at clean/s4/s8/s14 (the raw mean margin is
      confounded by the car-dominated test set: it keeps rising into the
      reversal; per-class averaging removes that).
  (4) PER-RANK spectral arrays (quant vs blur residual energy along the FP32
      SVD basis) + per-sigma subspace cosine {2,4,6,8,10,14} + rank-band sums,
      for the mechanism figure and the band decomposition.
Probes on seed 0 (matching the prior protocol); high-freq cells over 3 seeds.
Env: QAT_STEPS=400  HEAD_STEPS=3000  SEEDS=3  EPS=0.05
"""
import os
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
import numpy as np
np.float_ = np.float64; np.unicode_ = np.str_
import warnings; warnings.filterwarnings("ignore")
import copy, json, math
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

from dino_rotlsq_engine import (quantize_backbone, calibrate, qat_train, DinoClassifier,
                                DEV, MEAN, STD, BATCH)
from dino_crash_winwin import loader as tensor_loader
from dino_dawncrops_winwin import (build_crops, split_by_image, tensors, make_backbone,
                                   extract_feats, train_head_on_feats, eval_model,
                                   CLASSES, QAT_STEPS, HEAD_STEPS, SEEDS)

EPS = float(os.environ.get("EPS", "0.05"))
PERIODS = [112, 56, 28, 16, 8, 4]      # pixels/cycle; small period = high frequency
HF_CELLS = [("gaussian_noise", 3), ("gaussian_noise", 5),
            ("pixelate", 3), ("pixelate", 5)]
IMG = 224


def grating(period):
    xx = torch.arange(IMG).float()
    g = torch.sin(2 * math.pi * xx / period)
    return g.view(1, 1, 1, IMG).expand(1, 3, IMG, IMG)


@torch.inference_mode()
def sensitivity(model, imgs01, period):
    mean = torch.tensor(MEAN).view(1, 3, 1, 1).to(DEV)
    std = torch.tensor(STD).view(1, 3, 1, 1).to(DEV)
    g = grating(period).to(DEV) * EPS
    tot, n = 0.0, 0
    for i in range(0, imgs01.shape[0], BATCH):
        x = imgs01[i:i + BATCH].to(DEV)
        xn = (x - mean) / std
        xpn = ((x + g).clamp(0, 1) - mean) / std
        d = (model(xpn) - model(xn)).norm(dim=1)
        tot += d.sum().item(); n += x.shape[0]
    return tot / n


def main():
    print("=== DAWN-crop probe package 2 (freq-sens + high-freq controls) ===", flush=True)
    x, y, img_ids = build_crops()
    tr_idx, te_idx = split_by_image(y, img_ids)
    tr_xt = tensors(x[tr_idx]); tr_y = y[tr_idx]
    te_y = y[te_idx]
    # unnormalized [0,1] clean test crops for the grating probe
    te01 = torch.stack([T.ToTensor()(Image.fromarray(a)) for a in x[te_idx]])

    hf_xt = {f"{fam}_sev{v}": tensors(x[te_idx], corruption=fam, sev=v)
             for fam, v in HF_CELLS}
    clean_xt = tensors(x[te_idx])

    fp_backbone = make_backbone().to(DEV).eval()
    tr_feats = extract_feats(fp_backbone, tr_xt)

    out = {"freq_sens": {}, "hf_cells": {}}
    for s in range(SEEDS):
        torch.manual_seed(s); np.random.seed(s)
        head_c = train_head_on_feats(nn.Linear(384, len(CLASSES)), tr_feats, tr_y,
                                     HEAD_STEPS, batch=256)
        fp_head = train_head_on_feats(copy.deepcopy(head_c), tr_feats, tr_y,
                                      QAT_STEPS, batch=BATCH)
        fp = DinoClassifier(fp_backbone, fp_head).to(DEV).eval()
        qm = DinoClassifier(quantize_backbone(make_backbone()).to(DEV),
                            copy.deepcopy(head_c).to(DEV)).to(DEV)
        trl = tensor_loader(tr_xt, tr_y, True)
        calibrate(qm, trl); qat_train(qm, trl, QAT_STEPS)
        print(f"[seed {s}] trained", flush=True)

        for name, xt in list(hf_xt.items()) + [("clean", clean_xt)]:
            rf = eval_model(fp, xt, te_y); rq = eval_model(qm, xt, te_y)
            out["hf_cells"].setdefault(name, []).append(
                dict(fp32=rf["bal_acc"], w8a8=rq["bal_acc"]))
            print(f"[{name:>20} seed {s}] fp {rf['bal_acc']:.2f} q {rq['bal_acc']:.2f} "
                  f"({rq['bal_acc']-rf['bal_acc']:+.2f})", flush=True)

        if s == 0:
            for p in PERIODS:
                sf = sensitivity(fp, te01, p)
                sq = sensitivity(qm, te01, p)
                out["freq_sens"][str(p)] = dict(fp32=sf, w8a8=sq, ratio=sq / sf)
                print(f"[grating period {p:>4}px] fp {sf:.4f} q {sq:.4f} ratio {sq/sf:.3f}",
                      flush=True)
            probe_spectral_margin(fp, qm, x, te_idx, te_y, clean_xt, out)

    json.dump(out, open("dino_dawncrops_probes2.json", "w"), indent=2)
    print("\nDONE -> dino_dawncrops_probes2.json", flush=True)


@torch.inference_mode()
def _feats_logits(model, xt):
    F, L = [], []
    model.eval()
    for b in range(0, len(xt), BATCH):
        xx = xt[b:b+BATCH].to(DEV)
        f = model.backbone(xx)
        F.append(f.cpu()); L.append(model.head(f).cpu())
    return torch.cat(F).numpy(), torch.cat(L)


def _bal_margin(L, y):
    """Class-balanced mean margin: mean over classes of the class's mean margin."""
    yt = torch.as_tensor(y)
    true = L[torch.arange(len(yt)), yt]
    L2 = L.clone(); L2[torch.arange(len(yt)), yt] = -1e9
    m = true - L2.max(1).values
    per = [m[yt == c].mean().item() for c in range(len(CLASSES)) if (yt == c).sum() > 0]
    return float(np.mean(per))


def probe_spectral_margin(fp, qm, x, te_idx, te_y, clean_xt, out):
    F_fp_c, L_fp_c = _feats_logits(fp, clean_xt)
    F_q_c, L_q_c = _feats_logits(qm, clean_xt)
    Xc = F_fp_c - F_fp_c.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    quant = (F_q_c - F_fp_c) @ Vt.T
    eq = (quant**2).mean(0)
    sig = (Xc @ Vt.T)
    es = (sig**2).mean(0)
    out["spectral"] = {"quant_perrank": eq.tolist(), "signal_perrank": es.tolist(),
                       "cos_by_sigma": {}, "blur_perrank_s6": None,
                       "bands": {}}
    for sgm in (2, 4, 6, 8, 10, 14):
        xt = tensors(x[te_idx], sigma=float(sgm))
        F_b, _ = _feats_logits(fp, xt)
        blur = (F_b - F_fp_c) @ Vt.T
        eb = (blur**2).mean(0)
        cos = float(eq @ eb / (np.linalg.norm(eq) * np.linalg.norm(eb)))
        out["spectral"]["cos_by_sigma"][str(sgm)] = cos
        if sgm == 6:
            out["spectral"]["blur_perrank_s6"] = eb.tolist()
            k = 64
            out["spectral"]["bands"] = {
                "quant_low_rel": float(eq[:k].sum() / es[:k].sum()),
                "quant_tail_rel": float(eq[k:].sum() / es[k:].sum()),
                "blur_low_rel": float(eb[:k].sum() / es[:k].sum()),
                "blur_tail_rel": float(eb[k:].sum() / es[k:].sum())}
        print(f"[spectral sigma {sgm:>2}] cosine {cos:.3f}", flush=True)

    out["margin_balanced"] = {}
    for cell, sgm in [("clean", None), ("gauss_s4", 4.0), ("gauss_s8", 8.0),
                      ("gauss_s14", 14.0)]:
        xt = clean_xt if sgm is None else tensors(x[te_idx], sigma=sgm)
        _, Lf = _feats_logits(fp, xt)
        _, Lq = _feats_logits(qm, xt)
        out["margin_balanced"][cell] = _bal_margin(Lq, te_y) - _bal_margin(Lf, te_y)
        print(f"[bal margin {cell:>10}] gap {out['margin_balanced'][cell]:+.3f}", flush=True)


if __name__ == "__main__":
    main()
