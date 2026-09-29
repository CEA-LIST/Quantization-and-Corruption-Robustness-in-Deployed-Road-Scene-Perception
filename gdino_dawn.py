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

"""Grounding DINO zero-shot on DAWN: second-detector breadth point for the
detection paper. Open-vocabulary detection by text prompt, NO training, same
split / cells / COCO-mAP protocol as rtdetr_dawn.py.

Role in the paper: an independent detector family (Swin + BERT grounding vs
RT-DETR CNN-hybrid) evaluated on the identical testbed, so the FP32 blur
degradation profile of the testbed is not architecture-specific.
Env: MODEL=IDEA-Research/grounding-dino-tiny BATCH=4
"""
import os
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
import numpy as np
np.float_ = np.float64; np.unicode_ = np.str_
import warnings; warnings.filterwarnings("ignore")
import json
import torch
from PIL import Image

from rtdetr_dawn import load_dawn_det, split_by_image, render640, compute_map, CELLS, IMG

DEV = "cuda"
MODEL_ID = os.environ.get("MODEL", "IDEA-Research/grounding-dino-tiny")
BATCH = int(os.environ.get("BATCH", "4"))
SMOKE = int(os.environ.get("SMOKE", "0"))
# one phrase per DAWN class, period-separated as GDINO expects
PROMPT = "a car. a truck. a bus. a person."
PHRASE2CLS = {"car": 0, "truck": 1, "bus": 2, "person": 3}


def to_pil(item, **kw):
    x = render640(item, **kw)                      # [3,H,W] in [0,1], blur applied
    return Image.fromarray((x.permute(1, 2, 0).numpy() * 255).astype(np.uint8))


@torch.inference_mode()
def predict_cell_gdino(model, proc, items, **kw):
    model.eval()
    np.random.seed(4242)
    preds, targets = [], []
    for b0 in range(0, len(items), BATCH):
        chunk = items[b0:b0 + BATCH]
        ims = [to_pil(it, **kw) for it in chunk]
        inputs = proc(images=ims, text=[PROMPT] * len(ims), return_tensors="pt").to(DEV)
        out = model(**inputs)
        sizes = torch.tensor([[IMG, IMG]] * len(ims))
        res = proc.post_process_grounded_object_detection(
            out, inputs.input_ids, threshold=0.0, text_threshold=0.25,
            target_sizes=sizes)
        for it, r in zip(chunk, res):
            W, H = it["wh"]
            keep_b, keep_s, keep_l = [], [], []
            for box, sc, lab in zip(r["boxes"], r["scores"], r["text_labels"]):
                lab = lab.strip().lower()
                cls = next((c for p, c in PHRASE2CLS.items() if p in lab), None)
                if cls is None:
                    continue
                b = box.cpu() * torch.tensor([W / IMG, H / IMG, W / IMG, H / IMG])
                keep_b.append(b); keep_s.append(float(sc)); keep_l.append(cls)
            preds.append({
                "boxes": torch.stack(keep_b) if keep_b else torch.zeros(0, 4),
                "scores": torch.tensor(keep_s), "labels": torch.tensor(keep_l, dtype=torch.long)})
            n_gt, n_ig = len(it["boxes"]), len(it["ign_boxes"])
            targets.append({
                "boxes": torch.tensor(np.concatenate([it["boxes"], it["ign_boxes"]]),
                                      dtype=torch.float32),
                "labels": torch.tensor(np.concatenate([it["labels"], it["ign_labels"]]),
                                       dtype=torch.long),
                "iscrowd": torch.tensor([0] * n_gt + [1] * n_ig, dtype=torch.long)})
    return preds, targets


def main():
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    items = load_dawn_det()
    tr, te = split_by_image(items, seed=0)
    if SMOKE:
        te = te[:30]
    print(f"[data] test {len(te)} imgs | model {MODEL_ID}", flush=True)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(DEV)
    proc = AutoProcessor.from_pretrained(MODEL_ID)
    res = {}
    for name, kw in CELLS:
        p, t = predict_cell_gdino(model, proc, te, **kw)
        res[name] = compute_map(p, t)
        print(f"    {name:18s} mAP {res[name]['mAP']:.4f}  mAP50 {res[name]['mAP50']:.4f}",
              flush=True)
    json.dump({"model": MODEL_ID, "zs": res, "n_test": len(te)},
              open("gdino_dawn_zs.json", "w"), indent=2)
    print("wrote gdino_dawn_zs.json", flush=True)


if __name__ == "__main__":
    main()
