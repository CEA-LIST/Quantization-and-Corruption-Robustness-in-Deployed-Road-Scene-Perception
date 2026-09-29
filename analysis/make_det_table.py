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

"""Emit the LaTeX rows for the detection three-arm table from
rtdetr_dawn_rob.json: per cell, mAP mean+-std for FP32 / W8A8-QAT / PTQ and
the QAT-FP32 delta. Copy-paste target for paper/main.tex.
"""
import json
import sys
import numpy as np

import os
PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "rtdetr_dawn_rob.json")
S = json.load(open(PATH))["summary"]

LABEL = {
    "clean": "clean",
    "gauss_s1": "Gaussian $\\sigma{=}1$ (sev1)",
    "gauss_s2": "Gaussian $\\sigma{=}2$ (sev2)",
    "gauss_s3": "Gaussian $\\sigma{=}3$ (sev3)",
    "gauss_s4": "Gaussian $\\sigma{=}4$ (sev4)",
    "gauss_s6": "Gaussian $\\sigma{=}6$ (sev5)",
    "gauss_s8": "Gaussian $\\sigma{=}8$ (sev6)",
    "defocus_blur_sev3": "Defocus sev3",
    "defocus_blur_sev4": "Defocus sev4",
    "defocus_blur_sev5": "Defocus sev5",
    "motion_blur_sev3": "Motion sev3",
    "motion_blur_sev4": "Motion sev4",
    "motion_blur_sev5": "Motion sev5",
}


def cell(c, arm):
    v = np.array([r["mAP"] for r in S[c][arm]]) * 100
    return v.mean(), v.std()


for c in LABEL:
    if c not in S:
        continue
    f, fs = cell(c, "fp32")
    q, qs = cell(c, "w8a8")
    p, ps = cell(c, "ptq")
    print(f"{LABEL[c]} & ${f:.1f}${{\\scriptsize\\,$\\pm{fs:.1f}$}}"
          f" & ${q:.1f}${{\\scriptsize\\,$\\pm{qs:.1f}$}}"
          f" & ${p:.1f}${{\\scriptsize\\,$\\pm{ps:.1f}$}}"
          f" & ${q - f:+.1f}$ \\\\")
