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

"""Shared helper: the image corruption transform (Hendrycks & Dietterich
`imagecorruptions`) applied at evaluation time.

The module name is kept from the development history so the experiment scripts import
it unchanged.
"""

# Must precede `imagecorruptions` import (NumPy 2.0 removed np.float_/np.unicode_).
import numpy as np
np.float_ = np.float64
np.unicode_ = np.str_

import warnings

import torchvision

from imagecorruptions import corrupt


warnings.filterwarnings("ignore")

# ==========================================================
# CONFIG
# ==========================================================
IMG_SIZE = 224


# ==========================================================
# Corruption transform
# ==========================================================
class ApplyImageCorruption:
    """Apply Hendrycks `imagecorruptions.corrupt()` to a PIL image (post-resize)."""
    def __init__(self, corruption_name: str, severity: int):
        self.corruption_name = corruption_name
        self.severity = severity
    def __call__(self, img):
        arr = np.array(img)
        if arr.ndim == 2: arr = np.stack([arr]*3, axis=-1)
        if arr.shape[-1] == 4: arr = arr[..., :3]
        out = corrupt(arr, corruption_name=self.corruption_name, severity=self.severity)
        return torchvision.transforms.functional.to_pil_image(out)


