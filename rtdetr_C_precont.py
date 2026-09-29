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

"""Reviewer-2 control: blur sweep on the converged clean detector C, BEFORE the
FP32/QAT continuation stage. Evaluation only, no training. Identical data split,
cells, and metric as PHASE=ROB, so the numbers drop straight into Table I's frame.

Answers: does clean-DAWN fine-tuning specialize the model in a way that changes
the blur-robustness profile the quantization comparison is built on?
"""
import os, json
os.environ.setdefault("PHASE", "NONE")
from rtdetr_dawn import load_dawn_det, split_by_image, coco2dawn_map, eval_sweep, MODEL_ID, DEV

def main():
    from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
    import torch
    items = load_dawn_det()
    tr, te = split_by_image(items, seed=0)
    print(f"[data] train {len(tr)} | test {len(te)} imgs | model {MODEL_ID}", flush=True)

    model = RTDetrForObjectDetection.from_pretrained(MODEL_ID).to(DEV)
    proc = RTDetrImageProcessor.from_pretrained(MODEL_ID)
    cmap = coco2dawn_map(model)

    C = torch.load("rtdetr_dawn_C.pt", map_location="cpu")
    model.load_state_dict(C)
    model.eval()
    print("[C] converged clean detector, pre-continuation, blur sweep:", flush=True)
    res = eval_sweep(model, proc, te, cmap)
    json.dump({"model": MODEL_ID, "C_precont": res, "n_test": len(te)},
              open("rtdetr_C_precont.json", "w"), indent=2)
    print("[C] wrote rtdetr_C_precont.json", flush=True)

if __name__ == "__main__":
    main()
