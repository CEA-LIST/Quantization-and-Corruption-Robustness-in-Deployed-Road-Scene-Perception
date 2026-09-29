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

"""DAWN GT-box crops (detection dataset -> object classification): does the
clean-image W8A8 QAT win-win reproduce on REAL AV detection-dataset objects?

Rationale: the effect is established as a classification-level property; the user
requires detection datasets only. We therefore classify the GT-annotated road
objects of DAWN (adverse-weather AV detection benchmark, real bbox GT): classes
{Car, Truck, Bus, Person}, min box side 40px, square context crop, 224 bicubic.
"Clean" = the as-captured crop (no synthetic corruption); blur is injected at
EVALUATION ONLY; training never sees synthetic corruption.

PRE-REGISTERED PREDICTIONS (before running):
  P1  d(balanced acc) under gaussian blur follows the band: ~neutral at sigma 1-2,
      positive and growing through the severity 4-6 window (sigma 4/6/8).
  P2  FAMILY CONSISTENCY: defocus/motion sev 4-5 agree in sign with gaussian at a
      matched operating point (the check the crash task failed).
  P3  Clean balanced acc statistically matched (equalized-head protocol).
  Null or family-inconsistent -> reported as null.

Protocol (equalized heads, v2): head trained to convergence on frozen FP32
features (shared start C); FP32 = C + 400 no-quant continuation steps;
W8A8 = C + 400 QAT steps (LSQ scales + head). Same budgets; quantization is the
only difference. 3 seeds. Split is BY IMAGE (75/25), fixed across seeds; car
capped in TRAIN for balance; metrics: balanced acc (primary), plain acc, ECE, AURC.
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
import torch.nn.functional as Fn
import torchvision.transforms as T
from PIL import Image, ImageFilter
import timm

from dino_rotlsq_engine import (quantize_backbone, calibrate, qat_train, DinoClassifier,
                                DEV, MEAN, STD, BATCH)
from rot_lsq_corruptions_benchmark import ApplyImageCorruption, IMG_SIZE
from dino_crash_winwin import loader as tensor_loader

QAT_STEPS = int(os.environ.get("QAT_STEPS", "400"))
HEAD_STEPS = int(os.environ.get("HEAD_STEPS", "3000"))
SEEDS = int(os.environ.get("SEEDS", "3"))
CLASSES = ["Car", "Truck", "Bus", "Person"]
MIN_SIDE = 40
CAR_TRAIN_CAP = 700
_tt = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])

CELLS = ([("clean", None, 0, None)] +
         [(f"gauss_s{s:g}", None, 0, float(s)) for s in (1, 2, 3, 4, 6, 8)] +
         [(f"{fam}_sev{v}", fam, v, None) for fam in ("defocus_blur", "motion_blur")
          for v in (4, 5)])


def build_crops():
    from datasets import load_dataset, concatenate_datasets
    ds = concatenate_datasets([load_dataset("Maxim37/dawn-dataset", split="train"),
                               load_dataset("Maxim37/dawn-dataset", split="val")])
    xs, ys, img_ids = [], [], []
    for ex in ds:
        im = ex["image"].convert("RGB")
        W, H = im.size
        for o in ex["objects"]:
            if o["class_name"] not in CLASSES:
                continue
            w, h = o["width"], o["height"]
            if min(w, h) < MIN_SIDE:
                continue
            cx, cy = o["x_min"] + w / 2, o["y_min"] + h / 2
            side = max(w, h) * 1.1
            x0 = int(np.clip(cx - side / 2, 0, W - 1)); x1 = int(np.clip(cx + side / 2, 1, W))
            y0 = int(np.clip(cy - side / 2, 0, H - 1)); y1 = int(np.clip(cy + side / 2, 1, H))
            crop = im.crop((x0, y0, x1, y1)).resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
            xs.append(np.asarray(crop, dtype=np.uint8))
            ys.append(CLASSES.index(o["class_name"]))
            img_ids.append(ex["image_id"])
    return np.stack(xs), np.array(ys), np.array(img_ids)


def split_by_image(y, img_ids, frac_test=0.25, seed=0):
    rng = np.random.RandomState(seed)
    uniq = np.unique(img_ids); rng.shuffle(uniq)
    test_imgs = set(uniq[:int(len(uniq) * frac_test)])
    te = np.array([i in test_imgs for i in img_ids])
    tr_idx = np.where(~te)[0]; te_idx = np.where(te)[0]
    # cap the dominant car class in TRAIN only
    car = CLASSES.index("Car")
    car_tr = tr_idx[y[tr_idx] == car]; rng.shuffle(car_tr)
    keep = set(car_tr[:CAR_TRAIN_CAP]) | set(tr_idx[y[tr_idx] != car])
    tr_idx = np.array(sorted(keep))
    return tr_idx, te_idx


def tensors(x_u8, corruption=None, sev=0, sigma=None):
    out = []
    for a in x_u8:
        im = Image.fromarray(a)
        if sigma is not None and sigma > 0:
            im = im.filter(ImageFilter.GaussianBlur(radius=sigma))
        elif corruption is not None and sev > 0:
            im = ApplyImageCorruption(corruption, sev)(im)
        out.append(_tt(im))
    return torch.stack(out)


def make_backbone():
    return timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True,
                             num_classes=0, img_size=IMG_SIZE)


@torch.inference_mode()
def extract_feats(backbone, xt):
    backbone.eval(); F = []
    for b in range(0, len(xt), BATCH):
        F.append(backbone(xt[b:b+BATCH].to(DEV)).cpu())
    return torch.cat(F)


def train_head_on_feats(head, feats, y, steps, batch, lr=1e-3):
    head = head.to(DEV).train()
    opt = torch.optim.AdamW(head.parameters(), lr=lr)
    y = torch.as_tensor(y); n = len(y)
    for _ in range(steps):
        idx = torch.randint(0, n, (batch,))
        loss = Fn.cross_entropy(head(feats[idx].to(DEV)), y[idx].to(DEV))
        opt.zero_grad(); loss.backward(); opt.step()
    return head.eval()


def ece(probs, labels, n_bins=15):
    conf, pred = probs.max(1)
    conf = conf.numpy(); correct = (pred == labels).numpy().astype(float)
    bins = np.linspace(0, 1, n_bins + 1); e = 0.0
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum() > 0:
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return 100.0 * e


def aurc(probs, labels):
    conf, pred = probs.max(1)
    err = (pred != labels).numpy().astype(float)
    order = np.argsort(-conf.numpy())
    err = err[order]; risks = np.cumsum(err) / (np.arange(len(err)) + 1)
    return 1000.0 * risks.mean()


@torch.inference_mode()
def eval_model(model, xt, y):
    model.eval(); P = []
    for b in range(0, len(xt), BATCH):
        P.append(torch.softmax(model(xt[b:b+BATCH].to(DEV)), 1).cpu())
    p = torch.cat(P); yt = torch.as_tensor(y)
    pred = p.argmax(1)
    per_class = [float((pred[yt == c] == c).float().mean()) for c in range(len(CLASSES))
                 if (yt == c).sum() > 0]
    return dict(bal_acc=100.0 * float(np.mean(per_class)),
                acc=100.0 * float((pred == yt).float().mean()),
                ece=ece(p, yt), aurc=aurc(p, yt))


def main():
    print(f"=== DAWN GT-crop classification: FP32 vs clean-image W8A8 QAT, {SEEDS} seeds ===", flush=True)
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

    results = {name: {"fp32": [], "w8a8": []} for name in cells_xt}
    for name, xt in cells_xt.items():
        for s, fp, qm in models:
            rf = eval_model(fp, xt, te_y); rq = eval_model(qm, xt, te_y)
            results[name]["fp32"].append(rf); results[name]["w8a8"].append(rq)
            print(f"[{name:>16} seed {s}] bal {rf['bal_acc']:.2f}->{rq['bal_acc']:.2f} "
                  f"({rq['bal_acc']-rf['bal_acc']:+.2f}) | acc {rf['acc']:.2f}->{rq['acc']:.2f} "
                  f"({rq['acc']-rf['acc']:+.2f})", flush=True)

    summary = {}
    print(f"\n{'cell':>16}{'metric':>8}{'FP32':>14}{'W8A8':>14}{'delta':>9}")
    for name in results:
        summary[name] = {}
        for met in ("bal_acc", "acc", "ece", "aurc"):
            fv = [r[met] for r in results[name]["fp32"]]
            qv = [r[met] for r in results[name]["w8a8"]]
            d = float(np.mean(qv) - np.mean(fv))
            summary[name][met] = dict(fp32=[float(np.mean(fv)), float(np.std(fv))],
                                      w8a8=[float(np.mean(qv)), float(np.std(qv))],
                                      delta=d)
            print(f"{name:>16}{met:>8}{np.mean(fv):>9.2f}±{np.std(fv):<4.2f}"
                  f"{np.mean(qv):>9.2f}±{np.std(qv):<4.2f}{d:>+9.2f}", flush=True)

    json.dump({"summary": summary, "seeds": SEEDS, "classes": CLASSES,
               "n_train": int(len(tr_idx)), "n_test": int(len(te_idx)),
               "qat_steps": QAT_STEPS, "head_steps": HEAD_STEPS},
              open("dino_dawncrops_winwin.json", "w"), indent=2)
    print("\nDONE -> dino_dawncrops_winwin.json", flush=True)


if __name__ == "__main__":
    main()
