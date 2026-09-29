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

"""Shared helper for the DINOv2 recognition scripts: the tensor DataLoader.

The module name is kept from the development history so the experiment scripts import
it unchanged. Importing it seeds torch and NumPy with 0, as in the paper runs.
"""
import os
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
import numpy as np
np.float_ = np.float64; np.unicode_ = np.str_
import warnings; warnings.filterwarnings("ignore")
import torch

from dino_rotlsq_engine import BATCH

SEED = 0
torch.manual_seed(SEED); np.random.seed(SEED)


def loader(xt, yt, shuffle):
    ds = torch.utils.data.TensorDataset(xt, torch.as_tensor(yt))
    return torch.utils.data.DataLoader(ds, batch_size=BATCH, shuffle=shuffle)


