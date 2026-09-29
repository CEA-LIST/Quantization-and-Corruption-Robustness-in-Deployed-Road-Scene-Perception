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

"""Shared helper for the RT-DETR scripts: the DAWN class list and the by-image
train/test split.

The module name is kept from the development history so the experiment scripts import
it unchanged. Importing it disables timm's fused attention, as in the paper runs.
"""
import os
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
import numpy as np
np.float_ = np.float64; np.unicode_ = np.str_
import warnings; warnings.filterwarnings("ignore")

import timm
timm.layers.set_fused_attn(False)

CLASSES = ["Car", "Truck", "Bus", "Person"]


def split_by_image(items, frac_test=0.2, seed=0):
    ids = sorted({it["image_id"] for it in items})
    rng = np.random.RandomState(seed); rng.shuffle(ids)
    test = set(ids[:int(len(ids) * frac_test)])
    tr = [it for it in items if it["image_id"] not in test]
    te = [it for it in items if it["image_id"] in test]
    return tr, te


