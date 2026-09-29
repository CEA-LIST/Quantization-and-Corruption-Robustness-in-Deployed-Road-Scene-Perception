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

"""Shared helpers for the DINOv2 recognition scripts: the classifier wrapper, LSQ W8A8
quantization of the backbone Linear layers, activation calibration and QAT.

Importing it patches skimage's `gaussian` for `imagecorruptions` (the removed
`multichannel=` argument) and disables timm's fused attention, as in the paper runs.
"""
import os
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
import numpy as np
np.float_ = np.float64; np.unicode_ = np.str_
# skimage removed gaussian(multichannel=); imagecorruptions still calls it
import skimage.filters as _skf
_og = _skf.gaussian
def _gc(image, *a, multichannel=None, **k):
    if multichannel is not None and "channel_axis" not in k:
        k["channel_axis"] = -1 if multichannel else None
    return _og(image, *a, **k)
_skf.gaussian = _gc
import imagecorruptions.corruptions as _icc
_icc.gaussian = _gc

import warnings; warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from tqdm import tqdm

from quantization.modules import QuantizedLinearLayer, EnhancedFakeQuantizer

timm.layers.set_fused_attn(False)   # manual attention, as in the paper runs
DEV = "cuda"
MEAN = (0.485, 0.456, 0.406); STD = (0.229, 0.224, 0.225)
METHOD = "lsq"
BATCH = 32


class DinoClassifier(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head
    def forward(self, x):
        return self.head(self.backbone(x))


def quantize_backbone(backbone):
    cfg = {"method": METHOD, "weight_bits": 8, "act_bits": 8,
           "learning_rate": 1e-5, "lsq_learning_rate": 1e-4}
    n = 0
    for name, mod in list(backbone.named_modules()):
        if isinstance(mod, nn.Linear) and "blocks" in name:
            parent = backbone
            for p in name.split(".")[:-1]:
                parent = getattr(parent, p) if not p.isdigit() else parent[int(p)]
            setattr(parent, name.split(".")[-1], QuantizedLinearLayer(mod, cfg))
            n += 1
    print(f"  quantized {n} backbone Linears (method={METHOD})", flush=True)
    return backbone


@torch.inference_mode()
def calibrate(model, loader, n_batches=8):
    aqs = [m for m in model.modules() if isinstance(m, EnhancedFakeQuantizer) and not m.is_weight_quantizer]
    for m in aqs:
        m.calibration_mode = True; m.observer_enabled = True
    it = iter(loader)
    for _ in range(n_batches):
        x, _ = next(it)
        model(x.to(DEV))
    for m in aqs:
        m.calibration_mode = False
        m.init_learnable_scale()
    print(f"  calibrated {len(aqs)} activation quantizers", flush=True)


def qat_train(model, loader, steps):
    model.train()
    # Preserve DINOv2's pretrained features (freeze backbone weights). Train only
    # the LSQ activation scales (the quantization itself) + the linear head (so it
    # tracks the quantized features). Fine-tuning the backbone on small CIFAR-100
    # drifts the features away from the head and collapses accuracy.
    for p in model.parameters():
        p.requires_grad = False
    lsq = []
    for m in model.backbone.modules():
        if isinstance(m, EnhancedFakeQuantizer) and getattr(m, "learnable_scale", False):
            m.scale.requires_grad = True; lsq.append(m.scale)
    for p in model.head.parameters():
        p.requires_grad = True
    opt = torch.optim.AdamW([
        {"params": lsq, "lr": 5e-4},
        {"params": list(model.head.parameters()), "lr": 1e-3},
    ])
    print(f"  training {len(lsq)} LSQ scales + head (backbone frozen)", flush=True)
    it = iter(loader); done = 0
    pbar = tqdm(total=steps, desc="  QAT")
    while done < steps:
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(loader); x, y = next(it)
        x, y = x.to(DEV), y.to(DEV)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad(); loss.backward(); opt.step()
        done += 1; pbar.update(1)
        if done % 50 == 0:
            pbar.set_postfix(loss=f"{loss.item():.3f}")
    pbar.close()
    model.eval()


