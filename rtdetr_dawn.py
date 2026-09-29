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

"""RT-DETR on DAWN: pretrained real-time detection transformer as the paper's
detector (user directive: use established pretrained detectors, do not train
one from scratch).

Phases (env PHASE):
  ZS  : zero-shot eval of the COCO checkpoint on DAWN (class-mapped), clean +
        blur sweep. No training at all. Establishes the FP32 operating curve.
  FT  : fine-tune on CLEAN DAWN train images only -> converged clean detector C.
  ROB : from C, three matched-budget arms (equalized protocol):
          FP32 : C + M continuation steps (no quantizers)
          W8A8 : C + calibrate + M QAT steps (LSQ scales + trainable parts)
          PTQ  : C + calibrate only (no gradients)  [isolates QAT from INT8]
        Detection mAP under the blur sweep, 3 seeds.

Protocol notes:
  - Split BY IMAGE, seed 0, 80/20 (identical to dino_detr_dawn.py).
  - Blur is injected at the model input resolution (640x640) at EVALUATION
    ONLY; training never sees synthetic corruption.
  - Metric: COCO mAP[.5:.95] + mAP@50 via torchmetrics/pycocotools, macro over
    {Car, Truck, Bus, Person}; evaluated in the original image pixel frame.
  - COCO -> DAWN class map: person->Person, car->Car, bus->Bus, truck->Truck.
Env: MODEL=PekingU/rtdetr_r18vd IMG=640 BATCH=8 SEEDS=3 CONT_STEPS=600
"""
import os
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
import numpy as np
np.float_ = np.float64; np.unicode_ = np.str_
import warnings; warnings.filterwarnings("ignore")
import json, time
import torch
import torch.nn as nn
from PIL import Image, ImageFilter

from rot_lsq_corruptions_benchmark import ApplyImageCorruption
from dino_detr_dawn import split_by_image, CLASSES  # same split fn / class order
MIN_SIDE = 20

DEV = "cuda"
MODEL_ID = os.environ.get("MODEL", "PekingU/rtdetr_r18vd")
IMG = int(os.environ.get("IMG", "640"))
BATCH = int(os.environ.get("BATCH", "8"))
SEEDS = int(os.environ.get("SEEDS", "3"))
CONT_STEPS = int(os.environ.get("CONT_STEPS", "600"))
SMOKE = int(os.environ.get("SMOKE", "0"))

# COCO id -> DAWN class index
COCO2DAWN = {}


def load_dawn_det():
    """DAWN images with 4-class GT boxes. Identical image set and kept-box rule
    as dino_detr_dawn.load_dawn_det (so split_by_image gives the same split),
    plus ignore boxes: same 4 classes but min side < MIN_SIDE. Ignores enter the
    mAP evaluation as iscrowd (detections on them are neither TP nor FP) and are
    excluded from training targets."""
    from datasets import load_dataset, concatenate_datasets
    ds = concatenate_datasets([load_dataset("Maxim37/dawn-dataset", split="train"),
                               load_dataset("Maxim37/dawn-dataset", split="val")])
    out = []
    for ex in ds:
        im = ex["image"].convert("RGB"); W, H = im.size
        bs, ls, ib, il = [], [], [], []
        for o in ex["objects"]:
            if o["class_name"] not in CLASSES:
                continue
            w, h = o["width"], o["height"]
            box = [o["x_min"], o["y_min"], o["x_min"] + w, o["y_min"] + h]
            if min(w, h) < MIN_SIDE:
                ib.append(box); il.append(CLASSES.index(o["class_name"]))
            else:
                bs.append(box); ls.append(CLASSES.index(o["class_name"]))
        if not bs:
            continue
        out.append({"img": np.asarray(im, dtype=np.uint8), "wh": (W, H),
                    "boxes": np.array(bs, dtype=np.float32),
                    "labels": np.array(ls, dtype=np.int64),
                    "ign_boxes": np.array(ib, dtype=np.float32).reshape(-1, 4),
                    "ign_labels": np.array(il, dtype=np.int64),
                    "image_id": ex["image_id"]})
    return out


def coco2dawn_map(model):
    m = {}
    for cid, name in model.config.id2label.items():
        n = name.lower()
        if n == "car": m[int(cid)] = CLASSES.index("Car")
        elif n == "truck": m[int(cid)] = CLASSES.index("Truck")
        elif n == "bus": m[int(cid)] = CLASSES.index("Bus")
        elif n == "person": m[int(cid)] = CLASSES.index("Person")
    assert len(m) == 4, m
    return m


def render640(item, sigma=None, corruption=None, sev=0):
    """Original uint8 image -> blurred (eval-only) tensor [3,IMG,IMG] in [0,1]."""
    im = Image.fromarray(item["img"]).resize((IMG, IMG), Image.BICUBIC)
    if sigma is not None and sigma > 0:
        im = im.filter(ImageFilter.GaussianBlur(radius=sigma))
    elif corruption is not None and sev > 0:
        im = ApplyImageCorruption(corruption, sev)(im)
    x = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1)
    return x


CELLS = ([("clean", dict())] +
         [(f"gauss_s{s:g}", dict(sigma=float(s))) for s in (1, 2, 3, 4, 6, 8)] +
         [(f"{fam}_sev{v}", dict(corruption=fam, sev=v))
          for fam in ("defocus_blur", "motion_blur") for v in (3, 4, 5)])


@torch.inference_mode()
def predict_cell(model, proc, items, cmap, sigma=None, corruption=None, sev=0):
    """Returns torchmetrics-style preds+targets lists in ORIGINAL pixel frame.
    np RNG is re-seeded per cell so stochastic corruptions (motion-blur angle)
    are IDENTICAL across arms, seeds, and models. Small GT enters as iscrowd."""
    model.eval()
    np.random.seed(4242)          # deterministic corruption instances per cell
    preds, targets = [], []
    for b0 in range(0, len(items), BATCH):
        chunk = items[b0:b0 + BATCH]
        xs = torch.stack([render640(it, sigma, corruption, sev) for it in chunk]).to(DEV)
        out = model(pixel_values=xs)
        sizes = torch.tensor([[it["wh"][1], it["wh"][0]] for it in chunk])  # (H,W)
        res = proc.post_process_object_detection(out, target_sizes=sizes, threshold=0.0)
        for it, r in zip(chunk, res):
            keep = torch.tensor([int(l) in cmap for l in r["labels"]], dtype=torch.bool)
            lab = torch.tensor([cmap[int(l)] for l in r["labels"][keep]], dtype=torch.long)
            preds.append({"boxes": r["boxes"][keep].cpu(), "scores": r["scores"][keep].cpu(),
                          "labels": lab})
            n_gt, n_ig = len(it["boxes"]), len(it["ign_boxes"])
            targets.append({
                "boxes": torch.tensor(np.concatenate([it["boxes"], it["ign_boxes"]]),
                                      dtype=torch.float32),
                "labels": torch.tensor(np.concatenate([it["labels"], it["ign_labels"]]),
                                       dtype=torch.long),
                "iscrowd": torch.tensor([0] * n_gt + [1] * n_ig, dtype=torch.long)})
    return preds, targets


def compute_map(preds, targets):
    from torchmetrics.detection import MeanAveragePrecision
    mm = MeanAveragePrecision(box_format="xyxy", class_metrics=True)
    mm.update(preds, targets)
    r = mm.compute()
    per_cls = {}   # per-class mAP@[.5:.95] (NOT AP50)
    if "map_per_class" in r and r["map_per_class"].numel() > 1:
        for ci, c in zip(r["classes"].tolist(), r["map_per_class"].tolist()):
            per_cls[CLASSES[int(ci)]] = float(c)
    return {"mAP": float(r["map"]), "mAP50": float(r["map_50"]),
            "mAP75": float(r["map_75"]), "per_class_map": per_cls,
            "mar100": float(r["mar_100"])}


def eval_sweep(model, proc, items, cmap, cells=CELLS):
    res = {}
    for name, kw in cells:
        p, t = predict_cell(model, proc, items, cmap, **kw)
        res[name] = compute_map(p, t)
        print(f"    {name:18s} mAP {res[name]['mAP']:.4f}  mAP50 {res[name]['mAP50']:.4f}  "
              f"per-class {({k: round(v, 3) for k, v in res[name]['per_class_map'].items()})}",
              flush=True)
    return res


# --------------------------- training utilities -----------------------------
class FTSet(torch.utils.data.Dataset):
    """Clean-only training set. Labels in HF format (normalized cxcywh, COCO ids)."""
    def __init__(self, items, dawn2coco):
        self.items = items; self.d2c = dawn2coco
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        it = self.items[i]
        x = render640(it)                     # clean, never corrupted
        W, H = it["wh"]
        b = it["boxes"].copy()
        cx = (b[:, 0] + b[:, 2]) / 2 / W; cy = (b[:, 1] + b[:, 3]) / 2 / H
        ww = (b[:, 2] - b[:, 0]) / W; hh = (b[:, 3] - b[:, 1]) / H
        lab = {"class_labels": torch.tensor([self.d2c[int(l)] for l in it["labels"]],
                                            dtype=torch.long),
               "boxes": torch.tensor(np.stack([cx, cy, ww, hh], 1), dtype=torch.float32)}
        return x, lab


def collate_ft(batch):
    return torch.stack([b[0] for b in batch]), [b[1] for b in batch]


def freeze_bn(model):
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            m.eval()


def head_params(model):
    # Prediction heads only; the training-time denoising embedding is excluded.
    return [p for n, p in model.named_parameters()
            if ("class_embed" in n or "bbox_embed" in n) and "denoising" not in n]


SKIP_QUANT = ("class_embed", "bbox_embed", "enc_score_head", "enc_bbox_head",
              "denoising_class_embed", "query_pos_head")


def quantize_rtdetr(model):
    """W8A8 LSQ fake-quant on every Conv2d + Linear in the detector body
    (backbone + hybrid encoder + decoder). Prediction/query-selection heads
    stay FP32 (same 'FP32 task heads' convention as the rest of the paper)."""
    from quantization.modules import QuantizedLinearLayer, QuantizedConv2d
    cfg = {"method": "lsq", "weight_bits": 8, "act_bits": 8}
    n_lin = n_conv = 0
    for name, mod in list(model.named_modules()):
        if any(s in name for s in SKIP_QUANT):
            continue
        if isinstance(mod, (nn.Linear, nn.Conv2d)):
            parent = model
            parts = name.split(".")
            for p in parts[:-1]:
                parent = getattr(parent, p) if not p.isdigit() else parent[int(p)]
            if isinstance(mod, nn.Linear):
                setattr(parent, parts[-1], QuantizedLinearLayer(mod, cfg)); n_lin += 1
            else:
                setattr(parent, parts[-1], QuantizedConv2d(mod, cfg)); n_conv += 1
    print(f"  quantized {n_lin} Linear + {n_conv} Conv2d (LSQ W8A8), heads FP32", flush=True)
    return model


def calibrate_rtdetr(model, items, dawn2coco, n_batches=8):
    from quantization.modules import EnhancedFakeQuantizer
    aqs = [m for m in model.modules()
           if isinstance(m, EnhancedFakeQuantizer) and not m.is_weight_quantizer]
    for m in aqs:
        m.calibration_mode = True; m.observer_enabled = True
    dl = torch.utils.data.DataLoader(FTSet(items, dawn2coco), batch_size=BATCH,
                                     shuffle=True, collate_fn=collate_ft, num_workers=4)
    model.eval(); it = iter(dl)
    with torch.inference_mode():
        for _ in range(min(n_batches, len(dl))):
            xs, _ = next(it)
            model(pixel_values=xs.to(DEV))
    for m in aqs:
        m.calibration_mode = False; m.init_learnable_scale()
    print(f"  calibrated {len(aqs)} activation quantizers", flush=True)


def lsq_scales(model):
    from quantization.modules import EnhancedFakeQuantizer
    out = []
    for m in model.modules():
        if isinstance(m, EnhancedFakeQuantizer) and getattr(m, "learnable_scale", False):
            out.append(m.scale)
    return out


def train_steps(model, items, dawn2coco, steps, param_groups, log_every=100):
    """Matched-budget continuation: `steps` optimizer steps on CLEAN train images.
    BatchNorm running stats frozen so all arms keep C's statistics. No weight
    decay anywhere (LSQ scales must not be decayed; heads matched across arms)."""
    model.train(); freeze_bn(model)
    opt = torch.optim.AdamW(param_groups, weight_decay=0.0)
    dl = torch.utils.data.DataLoader(FTSet(items, dawn2coco), batch_size=BATCH,
                                     shuffle=True, collate_fn=collate_ft,
                                     num_workers=4, drop_last=True)
    it = iter(dl); done = 0; t0 = time.time()
    while done < steps:
        try: xs, lab = next(it)
        except StopIteration: it = iter(dl); xs, lab = next(it)
        lab = [{k: v.to(DEV) for k, v in l.items()} for l in lab]
        out = model(pixel_values=xs.to(DEV), labels=lab)
        opt.zero_grad(); out.loss.backward()
        torch.nn.utils.clip_grad_norm_([p for g in param_groups for p in g["params"]], 1.0)
        opt.step(); done += 1
        if done % log_every == 0:
            print(f"    step {done}/{steps} loss {out.loss.item():.3f} "
                  f"({(time.time()-t0)/done:.2f}s/it)", flush=True)
    model.eval()
    return model


def main():
    from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
    phase = os.environ.get("PHASE", "ZS")
    items = load_dawn_det()
    tr, te = split_by_image(items, seed=0)
    if SMOKE:
        tr = tr[:60]; te = te[:40]
    print(f"[data] train {len(tr)} imgs | test {len(te)} imgs | model {MODEL_ID}", flush=True)

    model = RTDetrForObjectDetection.from_pretrained(MODEL_ID).to(DEV)
    proc = RTDetrImageProcessor.from_pretrained(MODEL_ID)
    cmap = coco2dawn_map(model)
    dawn2coco = {v: k for k, v in cmap.items()}
    print(f"[map] COCO->DAWN {cmap}", flush=True)

    if phase == "ZS":
        print("[ZS] zero-shot COCO checkpoint on DAWN test:", flush=True)
        res = eval_sweep(model, proc, te, cmap)
        json.dump({"model": MODEL_ID, "zs": res, "n_test": len(te)},
                  open("rtdetr_dawn_zs.json", "w"), indent=2)
        print("[ZS] wrote rtdetr_dawn_zs.json", flush=True)
        return

    if phase == "FT":
        # Full-model fine-tune on CLEAN DAWN train -> converged clean detector C.
        # Model selection on a VALIDATION split carved out of train; the test
        # split te is never touched during FT.
        epochs = int(os.environ.get("EPOCHS", "40"))
        tr2, va = split_by_image(tr, frac_test=0.12, seed=1)
        print(f"[FT] train {len(tr2)} | val {len(va)} (test untouched)", flush=True)
        bb = [p for n, p in model.named_parameters() if "backbone" in n]
        rest = [p for n, p in model.named_parameters() if "backbone" not in n]
        opt = torch.optim.AdamW([{"params": bb, "lr": 1e-5},
                                 {"params": rest, "lr": 1e-4}], weight_decay=1e-4)
        dl = torch.utils.data.DataLoader(FTSet(tr2, dawn2coco), batch_size=BATCH,
                                         shuffle=True, collate_fn=collate_ft,
                                         num_workers=6, drop_last=True)
        total = epochs * len(dl)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, total)
        best = -1.0; step = 0; t0 = time.time()
        for ep in range(epochs):
            model.train()
            for xs, lab in dl:
                lab = [{k: v.to(DEV) for k, v in l.items()} for l in lab]
                out = model(pixel_values=xs.to(DEV), labels=lab)
                opt.zero_grad(); out.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); step += 1
                if step % 100 == 0:
                    print(f"  ep{ep} step{step}/{total} loss {out.loss.item():.3f} "
                          f"({(time.time()-t0)/step:.2f}s/it)", flush=True)
            if (ep + 1) % 5 == 0 or ep == epochs - 1:
                model.eval()
                p, t = predict_cell(model, proc, va, cmap)
                mm = compute_map(p, t)
                print(f"  [ep{ep+1}] VAL clean mAP {mm['mAP']:.4f} mAP50 {mm['mAP50']:.4f} "
                      f"per-class {({k: round(v,3) for k,v in mm['per_class_map'].items()})}",
                      flush=True)
                if mm["mAP"] > best:
                    best = mm["mAP"]
                    torch.save(model.state_dict(), "rtdetr_dawn_C.pt")
                    print(f"  [ep{ep+1}] saved rtdetr_dawn_C.pt (best VAL mAP {best:.4f})", flush=True)
        json.dump({"model": MODEL_ID, "best_val_map": best, "epochs": epochs,
                   "n_train": len(tr2), "n_val": len(va)},
                  open("rtdetr_dawn_ft.json", "w"), indent=2)
        return

    # PHASE == ROB : equalized three-arm protocol from the converged detector C.
    # Each arm re-seeds identically so data order, denoising noise, and
    # calibration batches match across arms within a seed.
    C = torch.load("rtdetr_dawn_C.pt", map_location="cpu")
    tr2, _va = split_by_image(tr, frac_test=0.12, seed=1)   # same pool FT trained on
    summary = {c: {"fp32": [], "w8a8": [], "ptq": []} for c, _ in CELLS}
    for seed in range(SEEDS):
        print(f"\n===== SEED {seed} =====", flush=True)
        def arm_seed():
            torch.manual_seed(1000 + seed); np.random.seed(1000 + seed)
        # ---- FP32 arm: heads-only continuation, no quantizers ----
        arm_seed()
        m = RTDetrForObjectDetection.from_pretrained(MODEL_ID).to(DEV)
        m.load_state_dict(C)
        for p in m.parameters(): p.requires_grad = False
        hp = head_params(m)
        for p in hp: p.requires_grad = True
        print(f"  [fp32] continuation: {sum(p.numel() for p in hp)/1e3:.0f}k head params", flush=True)
        train_steps(m, tr2, dawn2coco, CONT_STEPS, [{"params": hp, "lr": 1e-4}])
        r_fp = eval_sweep(m, proc, te, cmap); del m; torch.cuda.empty_cache()
        # ---- W8A8 QAT arm: quantize + calibrate + LSQ scales + heads ----
        arm_seed()
        m = RTDetrForObjectDetection.from_pretrained(MODEL_ID).to(DEV)
        m.load_state_dict(C)
        quantize_rtdetr(m); m.to(DEV)
        calibrate_rtdetr(m, tr2, dawn2coco)
        for p in m.parameters(): p.requires_grad = False
        hp = head_params(m); ls = lsq_scales(m)
        for p in hp + ls: p.requires_grad = True
        print(f"  [w8a8] QAT: {len(ls)} LSQ scales + heads", flush=True)
        train_steps(m, tr2, dawn2coco, CONT_STEPS,
                    [{"params": ls, "lr": 5e-4}, {"params": hp, "lr": 1e-4}])
        r_q = eval_sweep(m, proc, te, cmap)
        if seed == 0:
            torch.save(m.state_dict(), "rtdetr_dawn_w8a8_s0.pt")   # for mechanism probes
        del m; torch.cuda.empty_cache()
        # ---- PTQ arm: calibrate only (no gradient anywhere) ----
        arm_seed()
        m = RTDetrForObjectDetection.from_pretrained(MODEL_ID).to(DEV)
        m.load_state_dict(C)
        quantize_rtdetr(m); m.to(DEV)
        calibrate_rtdetr(m, tr2, dawn2coco)
        r_p = eval_sweep(m, proc, te, cmap); del m; torch.cuda.empty_cache()
        for c, _ in CELLS:
            summary[c]["fp32"].append(r_fp[c]); summary[c]["w8a8"].append(r_q[c])
            summary[c]["ptq"].append(r_p[c])
        json.dump({"model": MODEL_ID, "summary": summary, "seeds_done": seed + 1,
                   "cont_steps": CONT_STEPS, "n_test": len(te)},
                  open("rtdetr_dawn_rob.json", "w"), indent=2)
        print(f"[seed {seed}] wrote rtdetr_dawn_rob.json", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
