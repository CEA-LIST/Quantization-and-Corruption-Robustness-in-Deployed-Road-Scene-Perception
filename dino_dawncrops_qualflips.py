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

"""Qualitative flip mining for the DAWN GT-crop win-win: reproduce the exact
dino_dawncrops_winwin protocol (3 seeds, equalized heads, clean-only QAT) and
dump PER-SAMPLE predictions so we can show real crops where the FP32 control
assigns the WRONG class under blur while the W8A8 QAT model stays correct.

Selection is seed-majority (>=2/3, preferring 3/3) so no example is one-seed
noise, and the reverse-flip counts (W8A8 wrong, FP32 right) are recorded per
cell so the figure caption can state both directions honestly.

Outputs:
  dino_qualflips.json  per-cell flip counts + ranked candidate metadata
  dino_qualflips.npz   uint8 clean + corrupted crops for the candidates
"""
import os
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
import numpy as np
np.float_ = np.float64; np.unicode_ = np.str_
import warnings; warnings.filterwarnings("ignore")
import copy, json
import torch
import torch.nn as nn
from PIL import Image, ImageFilter

from dino_rotlsq_engine import (quantize_backbone, calibrate, qat_train,
                                DinoClassifier, DEV, BATCH)
from rot_lsq_corruptions_benchmark import ApplyImageCorruption
from dino_crash_winwin import loader as tensor_loader
from dino_dawncrops_winwin import (build_crops, split_by_image, tensors,
                                   make_backbone, extract_feats,
                                   train_head_on_feats, CLASSES,
                                   QAT_STEPS, HEAD_STEPS, SEEDS)

CELLS = [("clean", None, 0, None),
         ("gauss_s4", None, 0, 4.0),
         ("gauss_s6", None, 0, 6.0),
         ("gauss_s8", None, 0, 8.0),
         ("defocus_blur_sev4", "defocus_blur", 4, None),
         ("defocus_blur_sev5", "defocus_blur", 5, None)]
N_CAND = 40


@torch.inference_mode()
def preds(model, xt):
    model.eval(); P = []
    for b in range(0, len(xt), BATCH):
        P.append(model(xt[b:b+BATCH].to(DEV)).argmax(1).cpu())
    return torch.cat(P).numpy()


def corrupt_u8(a, fam, sev, sigma):
    im = Image.fromarray(a)
    if sigma is not None and sigma > 0:
        im = im.filter(ImageFilter.GaussianBlur(radius=sigma))
    elif fam is not None and sev > 0:
        im = ApplyImageCorruption(fam, sev)(im)
    return np.asarray(im, dtype=np.uint8)


def main():
    print(f"=== qualitative flip mining: winwin protocol, {SEEDS} seeds ===", flush=True)
    x, y, img_ids = build_crops()
    tr_idx, te_idx = split_by_image(y, img_ids)
    print(f"crops {len(y)} | train {len(tr_idx)} | test {len(te_idx)}", flush=True)

    tr_xt = tensors(x[tr_idx]); tr_y = y[tr_idx]
    te_y = y[te_idx]

    fp_backbone = make_backbone().to(DEV).eval()
    tr_feats = extract_feats(fp_backbone, tr_xt)

    models = []
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
        models.append((s, fp, qm))
        print(f"[seed {s}] trained", flush=True)

    # per-sample predictions, one cell at a time (materialize lazily)
    P = {}   # P[cell][arm] -> (SEEDS, n_test) int array
    for name, fam, sev, sigma in CELLS:
        xt = tensors(x[te_idx], corruption=fam, sev=sev, sigma=sigma)
        P[name] = {"fp32": [], "w8a8": []}
        for s, fp, qm in models:
            pf, pq = preds(fp, xt), preds(qm, xt)
            P[name]["fp32"].append(pf); P[name]["w8a8"].append(pq)
            bf = np.mean([(pf[te_y == c] == c).mean() for c in range(4)]) * 100
            bq = np.mean([(pq[te_y == c] == c).mean() for c in range(4)]) * 100
            print(f"[{name:>18} seed {s}] bal {bf:.2f}->{bq:.2f} ({bq-bf:+.2f})", flush=True)
        P[name] = {k: np.stack(v) for k, v in P[name].items()}
        del xt

    corr_n = {name: {a: (P[name][a] == te_y[None]).sum(0) for a in ("fp32", "w8a8")}
              for name, *_ in [c for c in CELLS]}

    # mine: clean-correct both arms (majority), corrupted FP32-wrong & W8A8-right
    maj = SEEDS // 2 + 1
    clean_ok = (corr_n["clean"]["fp32"] >= maj) & (corr_n["clean"]["w8a8"] >= maj)
    report, candidates = {}, []
    for name, fam, sev, sigma in CELLS[1:]:
        fp_w = SEEDS - corr_n[name]["fp32"]          # seeds where fp32 wrong
        q_c = corr_n[name]["w8a8"]                   # seeds where w8a8 correct
        fwd = clean_ok & (fp_w >= maj) & (q_c >= maj)          # fp32 wrong, qat right
        rev = clean_ok & (SEEDS - q_c >= maj) & (corr_n[name]["fp32"] >= maj)
        report[name] = {"fp32_wrong_qat_right": int(fwd.sum()),
                        "qat_wrong_fp32_right": int(rev.sum()),
                        "by_class_fwd": {CLASSES[c]: int((fwd & (te_y == c)).sum())
                                         for c in range(4)}}
        print(f"[{name:>18}] fp32-wrong/qat-right {fwd.sum():>3}  "
              f"reverse {rev.sum():>3}  net {int(fwd.sum())-int(rev.sum()):+d}", flush=True)
        for i in np.where(fwd)[0]:
            fp_preds = P[name]["fp32"][:, i]
            wrong = np.bincount(fp_preds[fp_preds != te_y[i]], minlength=4).argmax()
            candidates.append(dict(
                cell=name, idx=int(i), gt=CLASSES[te_y[i]],
                fp32_pred=CLASSES[int(wrong)],
                fp32_wrong_seeds=int(fp_w[i]), qat_right_seeds=int(q_c[i]),
                clean_ok_seeds=int(min(corr_n["clean"]["fp32"][i],
                                       corr_n["clean"]["w8a8"][i])),
                image_id=str(np.asarray(img_ids)[te_idx][i])))

    # rank: fully consistent first, minority classes first, then cell severity
    order = {c: r for r, c in enumerate(["Person", "Bus", "Truck", "Car"])}
    candidates.sort(key=lambda d: (-(d["fp32_wrong_seeds"] + d["qat_right_seeds"]
                                     + d["clean_ok_seeds"]),
                                   order[d["gt"]], d["cell"]))
    candidates = candidates[:N_CAND]

    cell_spec = {name: (fam, sev, sigma) for name, fam, sev, sigma in CELLS}
    clean_imgs = np.stack([x[te_idx][d["idx"]] for d in candidates]) if candidates else np.zeros((0,))
    corr_imgs = np.stack([corrupt_u8(x[te_idx][d["idx"]], *cell_spec[d["cell"]])
                          for d in candidates]) if candidates else np.zeros((0,))
    np.savez_compressed("dino_qualflips.npz", clean=clean_imgs, corrupted=corr_imgs)
    json.dump({"counts": report, "candidates": candidates, "seeds": SEEDS,
               "classes": CLASSES, "n_test": int(len(te_idx))},
              open("dino_qualflips.json", "w"), indent=2)
    print(f"\nDONE -> dino_qualflips.json ({len(candidates)} candidates) + dino_qualflips.npz",
          flush=True)


if __name__ == "__main__":
    main()
