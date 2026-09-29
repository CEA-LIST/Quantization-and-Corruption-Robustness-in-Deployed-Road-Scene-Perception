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

"""Review-driven follow-ups on the DAWN GT-crop recognition result.

Reproduces the EXACT dino_dawncrops_winwin.py protocol (same split seed, same
per-seed training sequence, same cells), then adds the two analyses both
review panels asked for:

(A) BOOTSTRAP CIs: per-crop predictions are saved for FP32 and W8A8 at every
    cell; a paired, class-stratified bootstrap over test crops (B=10000, same
    resample indices for both models and all seeds) gives a 95% CI and
    p(delta<=0) for the balanced-accuracy delta of each cell, plus per-class
    flip counts (how many crops each model gets exclusively right).

(B) NOISE/CONTRAST SPECTRAL CONTROL (the control that exposed the detector
    probe, now on recognition features, seed 0): per-rank residual profiles
    for white noise and contrast jitter at input magnitudes whose FEATURE
    residual energy brackets the blur ladder; profile cosine vs the quant
    residual for each, compared to blur's cosine at matched energy; plus a
    shuffled-rank null (cosine of quant profile vs 200 permutations of each
    perturbation profile).

Writes dino_dawncrops_reviewfixes.json.
"""
import os
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
import numpy as np
np.float_ = np.float64; np.unicode_ = np.str_
import warnings; warnings.filterwarnings("ignore")
import copy, json, collections
import torch
import torch.nn as nn
from PIL import Image, ImageEnhance

from dino_rotlsq_engine import (quantize_backbone, calibrate, qat_train,
                                DinoClassifier, DEV, BATCH)
from dino_crash_winwin import loader as tensor_loader
from dino_dawncrops_winwin import (build_crops, split_by_image, tensors,
                                   make_backbone, extract_feats,
                                   train_head_on_feats, CLASSES, CELLS,
                                   QAT_STEPS, HEAD_STEPS, SEEDS, _tt)

B_BOOT = 10000
NOISE_STDS = (0.02, 0.05, 0.10, 0.20, 0.40)
CONTRAST_FACTORS = (0.9, 0.8, 0.6, 0.4, 0.2)
BLUR_SIGMAS = (2, 4, 6, 8, 10, 14)
N_PERM = 200


def tensors_noise(x_u8, std, seed=0):
    rng = np.random.RandomState(seed)
    out = []
    for a in x_u8:
        v = a.astype(np.float32) / 255.0
        v = np.clip(v + rng.randn(*v.shape).astype(np.float32) * std, 0, 1)
        out.append(_tt(Image.fromarray((v * 255).astype(np.uint8))))
    return torch.stack(out)


def tensors_contrast(x_u8, factor):
    out = []
    for a in x_u8:
        out.append(_tt(ImageEnhance.Contrast(Image.fromarray(a)).enhance(factor)))
    return torch.stack(out)


@torch.inference_mode()
def preds(model, xt, y):
    model.eval(); P = []
    for b in range(0, len(xt), BATCH):
        P.append(model(xt[b:b+BATCH].to(DEV)).argmax(1).cpu())
    return (torch.cat(P) == torch.as_tensor(y)).numpy().astype(np.uint8)


@torch.inference_mode()
def feats(model_backbone, xt):
    model_backbone.eval(); F = []
    for b in range(0, len(xt), BATCH):
        F.append(model_backbone(xt[b:b+BATCH].to(DEV)).cpu())
    return torch.cat(F).numpy()


def profile(F_pert, F_clean, Vt):
    r = (F_pert - F_clean) @ Vt.T
    return (r ** 2).mean(0)


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    print(f"=== review fixes: bootstrap CIs + noise control ({SEEDS} seeds) ===",
          flush=True)
    x, y, img_ids = build_crops()
    tr_idx, te_idx = split_by_image(y, img_ids)
    print(f"crops {len(y)} | train {len(tr_idx)} "
          f"{collections.Counter(y[tr_idx].tolist())} | test {len(te_idx)} "
          f"{collections.Counter(y[te_idx].tolist())}", flush=True)

    tr_xt = tensors(x[tr_idx]); tr_y = y[tr_idx]
    te_y = y[te_idx]

    fp_backbone = make_backbone().to(DEV).eval()
    tr_feats = extract_feats(fp_backbone, tr_xt)

    models = []
    for s in range(SEEDS):
        torch.manual_seed(s); np.random.seed(s)
        head_c = train_head_on_feats(nn.Linear(384, len(CLASSES)), tr_feats,
                                     tr_y, HEAD_STEPS, batch=256)
        fp_head = train_head_on_feats(copy.deepcopy(head_c), tr_feats, tr_y,
                                      QAT_STEPS, batch=BATCH)
        fp = DinoClassifier(fp_backbone, fp_head).to(DEV).eval()
        qm = DinoClassifier(quantize_backbone(make_backbone()).to(DEV),
                            copy.deepcopy(head_c).to(DEV)).to(DEV)
        trl = tensor_loader(tr_xt, tr_y, True)
        calibrate(qm, trl); qat_train(qm, trl, QAT_STEPS)
        models.append((s, fp, qm))
        print(f"[seed {s}] trained", flush=True)
    del tr_xt, tr_feats

    # ---------- (A) per-crop predictions + paired stratified bootstrap ------
    rng = np.random.RandomState(0)
    cls_idx = [np.where(te_y == c)[0] for c in range(len(CLASSES))]
    boot_idx = [ci[rng.randint(0, len(ci), size=(B_BOOT, len(ci)))]
                for ci in cls_idx]

    boot = {}
    for name, fam, sev, sigma in CELLS:
        xt = tensors(x[te_idx], corruption=fam, sev=sev, sigma=sigma)
        d_seeds, flips, deltas_pt = [], [], []
        for s, fp, qm in models:
            cf = preds(fp, xt, te_y); cq = preds(qm, xt, te_y)
            rec_f = np.stack([cf[bi].mean(1) for bi in boot_idx])   # (4, B)
            rec_q = np.stack([cq[bi].mean(1) for bi in boot_idx])
            d_seeds.append(100.0 * (rec_q.mean(0) - rec_f.mean(0)))  # (B,)
            deltas_pt.append(100.0 * float(
                np.mean([cq[ci].mean() - cf[ci].mean() for ci in cls_idx])))
            flips.append({CLASSES[c]: dict(
                fp_only=int(((cf[ci] == 1) & (cq[ci] == 0)).sum()),
                q_only=int(((cf[ci] == 0) & (cq[ci] == 1)).sum()),
                n=int(len(ci)))
                for c, ci in enumerate(cls_idx)})
        d = np.mean(d_seeds, axis=0)                                # (B,)
        lo, hi = np.percentile(d, [2.5, 97.5])
        p_le0 = float((d <= 0).mean())
        boot[name] = dict(delta_point=float(np.mean(deltas_pt)),
                          ci95=[float(lo), float(hi)], p_delta_le_0=p_le0,
                          per_seed_delta=[float(v) for v in deltas_pt],
                          flips=flips)
        print(f"[boot {name:>16}] dBal {np.mean(deltas_pt):+5.2f} "
              f"CI95 [{lo:+5.2f},{hi:+5.2f}]  p(d<=0)={p_le0:.4f}", flush=True)
        del xt

    # ---------- (B) spectral noise/contrast control on seed-0 features ------
    s0, fp0, qm0 = models[0]
    clean_xt = tensors(x[te_idx])
    F_fp_c = feats(fp0.backbone, clean_xt)
    F_q_c = feats(qm0.backbone, clean_xt)
    Xc = F_fp_c - F_fp_c.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    eq = profile(F_q_c, F_fp_c, Vt)
    del clean_xt

    def perm_null(ep):
        prng = np.random.RandomState(1)
        cs = [cosine(eq, prng.permutation(ep)) for _ in range(N_PERM)]
        return [float(np.mean(cs)), float(np.std(cs))]

    spec = {"quant_energy": float(eq.sum()), "blur": {}, "noise": {},
            "contrast": {}}
    for sgm in BLUR_SIGMAS:
        eb = profile(feats(fp0.backbone, tensors(x[te_idx], sigma=float(sgm))),
                     F_fp_c, Vt)
        spec["blur"][str(sgm)] = dict(cos=cosine(eq, eb),
                                      energy=float(eb.sum()),
                                      null=perm_null(eb))
        print(f"[spectral blur s{sgm:>2}] cos {spec['blur'][str(sgm)]['cos']:.3f} "
              f"energy {eb.sum():.1f} null {spec['blur'][str(sgm)]['null'][0]:.3f}",
              flush=True)
    for std in NOISE_STDS:
        en = profile(feats(fp0.backbone, tensors_noise(x[te_idx], std)),
                     F_fp_c, Vt)
        spec["noise"][str(std)] = dict(cos=cosine(eq, en),
                                       energy=float(en.sum()),
                                       null=perm_null(en))
        print(f"[spectral noise {std:.2f}] cos {spec['noise'][str(std)]['cos']:.3f} "
              f"energy {en.sum():.1f} null {spec['noise'][str(std)]['null'][0]:.3f}",
              flush=True)
    for f in CONTRAST_FACTORS:
        ec = profile(feats(fp0.backbone, tensors_contrast(x[te_idx], f)),
                     F_fp_c, Vt)
        spec["contrast"][str(f)] = dict(cos=cosine(eq, ec),
                                        energy=float(ec.sum()),
                                        null=perm_null(ec))
        print(f"[spectral contrast {f:.1f}] cos {spec['contrast'][str(f)]['cos']:.3f} "
              f"energy {ec.sum():.1f} null {spec['contrast'][str(f)]['null'][0]:.3f}",
              flush=True)

    json.dump({"bootstrap": boot, "spectral_control": spec,
               "B_boot": B_BOOT, "n_perm": N_PERM, "seeds": SEEDS,
               "classes": CLASSES, "n_test": int(len(te_idx))},
              open("dino_dawncrops_reviewfixes.json", "w"), indent=2)
    print("\nDONE -> dino_dawncrops_reviewfixes.json", flush=True)


if __name__ == "__main__":
    main()
