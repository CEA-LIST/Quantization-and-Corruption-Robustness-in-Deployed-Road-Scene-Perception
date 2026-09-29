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

# ls_ood_detect_cea/quantization/modules.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Optional, Tuple


# --- Helpers ---
def get_bit_config(quant_cfg, default=8):
    w_bits = quant_cfg.get('weight_bits', quant_cfg.get('number_of_bits', default))
    a_bits = quant_cfg.get('act_bits', quant_cfg.get('number_of_bits', default))
    return w_bits, a_bits


# --- LSQ Implementation (Updated from Reference) ---
class LSQ(torch.autograd.Function):
    """
    Implements the LSQ (Learned Step-size Quantization) logic as a custom
    autograd function.
    """

    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        scale: torch.Tensor,
        q_min: int,
        q_max: int
    ) -> torch.Tensor:
        x_div_s = x / scale
        x_quant = torch.clamp(x_div_s, q_min, q_max).round()
        x_dequant = x_quant * scale
        
        ctx.save_for_backward(x, scale, x_quant)
        ctx.q_min, ctx.q_max = q_min, q_max
        return x_dequant

    @staticmethod
    def backward(
        ctx: Any, 
        grad_output: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], ...]:
        x, scale, x_quant = ctx.saved_tensors
        q_min, q_max = ctx.q_min, ctx.q_max

        # Gradient for input 'x' (Straight-Through Estimator)
        x_div_s = x / scale
        in_range_mask = (x_div_s >= q_min) & (x_div_s <= q_max)
        grad_x = torch.where(
            in_range_mask, grad_output, torch.zeros_like(grad_output)
        )

        # Gradient for the learnable 'scale'
        grad_scale_in_range = grad_output * (x_quant - x_div_s)
        grad_scale_out_of_range = grad_output * x_quant
        grad_scale_map = torch.where(
            in_range_mask, grad_scale_in_range, grad_scale_out_of_range
        )
        
        # Ensure correct reduction for broadcasting
        dims_to_sum = [
            i for i, (dx, ds) in enumerate(zip(x.shape, scale.shape)) if dx != ds
        ]
        if x.ndim > scale.ndim:
            dims_to_sum.extend(range(scale.ndim, x.ndim))
        
        grad_scale = grad_scale_map.sum(
            dim=tuple(set(dims_to_sum))
        ).reshape(scale.shape)
        
        # LSQ normalization term (Fixed as per reference)
        grad_scale /= math.sqrt(x.numel() * q_max)

        return grad_x, grad_scale, None, None

# --- Enhanced Fake Quantizer (Updated from Reference) ---
class EnhancedFakeQuantizer(nn.Module):
    """
    A versatile fake quantizer module for weights and activations.
    """
    def __init__(
        self,
        bits: int = 8,
        observer_momentum: float = 0.1,
        is_weight_quantizer: bool = False,
        per_channel: bool = False,
        num_channels: Optional[int] = None,
        learnable_scale: bool = False
    ):
        super().__init__()
        self.bits = bits
        self.is_weight_quantizer = is_weight_quantizer
        self.per_channel = per_channel and self.is_weight_quantizer
        # Reference logic: learnable scale primarily for activations
        self.learnable_scale = learnable_scale and not self.is_weight_quantizer

        if self.is_weight_quantizer:
            self.num_channels = num_channels if self.per_channel else 1
        else: # Activation quantizer
            self.num_channels = 1
            self.observer_momentum = observer_momentum
            self.register_buffer('min_val', torch.full((1,), float('inf')))
            self.register_buffer('max_val', torch.full((1,), float('-inf')))
            
            if self.learnable_scale:
                self.scale = nn.Parameter(torch.ones(1))
            else:
                self.register_buffer('scale', torch.ones(1))
            
            self.register_buffer('initialized', torch.tensor(False, dtype=torch.bool))
            self.calibration_mode = False

        # Compatibility for existing observers from previous implementation
        self.observer_enabled = True

    def disable_observer_update(self):
        self.observer_enabled = False

    def _get_qmin_qmax(self) -> Tuple[int, int]:
        q_min = -(2**(self.bits - 1))
        q_max = (2**(self.bits - 1)) - 1
        return q_min, q_max

    @torch.no_grad()
    def update_observer_stats(self, x: torch.Tensor) -> None:
        x_detached = x.detach().float()
        current_min = torch.min(x_detached)
        current_max = torch.max(x_detached)
        
        if not self.initialized.item():
            self.min_val.copy_(current_min)
            self.max_val.copy_(current_max)
            self.initialized.fill_(True)
        else:
            self.min_val.mul_(1 - self.observer_momentum).add_(
                current_min * self.observer_momentum
            )
            self.max_val.mul_(1 - self.observer_momentum).add_(
                current_max * self.observer_momentum
            )

    @torch.no_grad()
    def update_qparams(self) -> None:
        if self.learnable_scale:
            return
        max_abs = torch.max(torch.abs(self.min_val), torch.abs(self.max_val))
        _, q_max = self._get_qmin_qmax()
        self.scale.data.copy_((max_abs / q_max).clamp(min=1e-8))

    @torch.no_grad()
    def init_learnable_scale(self) -> None:
        if not self.learnable_scale or not self.initialized.item():
            return
        max_abs = torch.max(torch.abs(self.min_val), torch.abs(self.max_val))
        _, q_max = self._get_qmin_qmax()
        self.scale.data.copy_((max_abs / q_max).clamp(min=1e-8))

    # Added for compatibility with existing modules.py structure
    @torch.no_grad()
    def init_weight_scale(self, x):
        if self.is_weight_quantizer and not self.learnable_scale:
            # We perform immediate calculation for weights as they are often static in PTQ
            _, q_max = self._get_qmin_qmax()
            if self.per_channel:
                dims_to_reduce = tuple(range(1, x.ndim))
                max_abs = x.abs().amax(dim=dims_to_reduce, keepdim=True)
            else:
                max_abs = x.abs().max()
            # Note: In this reference impl, weights don't usually use a stored 'scale' buffer 
            # for PTQ, they calc on fly. But for consistency with external calls:
            pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bits is None:
            return x

        q_min, q_max = self._get_qmin_qmax()
        
        if self.is_weight_quantizer:
            if self.per_channel:
                dims_to_reduce = tuple(range(1, x.ndim))
                max_abs = x.abs().amax(dim=dims_to_reduce, keepdim=True)
            else:
                max_abs = x.abs().max()
            
            current_scale = (max_abs / q_max).clamp(min=1e-8)
            x_dequant = torch.clamp(torch.round(x / current_scale), q_min, q_max) * current_scale
            # Use Straight-Through Estimator (STE) for gradients
            return x + (x_dequant - x).detach()

        # Logic for activation quantization
        if self.calibration_mode:
            # Observe-only: collecting stats must not quantize, otherwise the
            # uninitialized scale (1.0) distorts every downstream observer.
            self.update_observer_stats(x)
            return x
        should_update = (self.training and not self.learnable_scale and self.observer_enabled)
        if should_update:
            self.update_observer_stats(x)
        
        if self.training and not self.learnable_scale and self.observer_enabled:
            self.update_qparams()
        
        if not self.initialized.item():
            return x
        
        if self.learnable_scale and self.training:
            # Use LSQ autograd function for training
            return LSQ.apply(x, self.scale.clamp(min=1e-8), q_min, q_max)
        
        # Standard PTQ/QAT inference path (clamped like the LSQ training path)
        current_scale = self.scale.float().clamp(min=1e-8)
        x_dequant = torch.clamp(torch.round(x / current_scale), q_min, q_max) * current_scale
        return x + (x_dequant - x).detach()

# --- Quantized Linear Layer (Updated) ---
class QuantizedLinearLayer(nn.Module):
    def __init__(self, original_linear_layer: nn.Linear, quant_cfg: Dict[str, Any]):
        super().__init__()
        self.original_linear_layer = original_linear_layer
        self.quant_cfg = quant_cfg
        
        # --- CONFIGURATION FLAGS ---
        method = quant_cfg.get('method', 'ptq')
        is_lsq = 'lsq' in method

        w_bits, a_bits = get_bit_config(quant_cfg)

        # --- WEIGHT QUANTIZER ---
        self.weight_quantizer = EnhancedFakeQuantizer(
            bits=w_bits,
            is_weight_quantizer=True,
            per_channel=True,
            num_channels=original_linear_layer.out_features,
            learnable_scale=is_lsq
        )
        
        # --- ACTIVATION QUANTIZER ---
        self.activation_quantizer = EnhancedFakeQuantizer(
            bits=a_bits,
            is_weight_quantizer=False,
            learnable_scale=is_lsq
        )

    # --- EXPOSE ORIGINAL WEIGHTS (Safe properties) ---
    @property
    def weight(self):
        return self.original_linear_layer.weight

    @property
    def bias(self):
        return self.original_linear_layer.bias

    def __getattr__(self, name):
        """Pass through attributes like 'in_features' to the inner layer."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.original_linear_layer, name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Quantization
        qx = self.activation_quantizer(x)
        qw = self.weight_quantizer(self.original_linear_layer.weight)
        
        # 2. Main Linear Op
        return F.linear(qx, qw, self.original_linear_layer.bias)

class QuantizedConv2d(nn.Module):
    def __init__(self, original_conv_layer, quant_cfg):
        super().__init__()
        self.original_conv_layer = original_conv_layer
        self.quant_cfg = quant_cfg
        w_bits, a_bits = get_bit_config(quant_cfg)
        is_lsq = 'lsq' in quant_cfg.get('method', '')

        self.weight_quantizer = EnhancedFakeQuantizer(
            bits=w_bits, is_weight_quantizer=True, per_channel=True, 
            num_channels=original_conv_layer.out_channels, learnable_scale=is_lsq
        )
        
        self.activation_quantizer = EnhancedFakeQuantizer(
            bits=a_bits, is_weight_quantizer=False, learnable_scale=is_lsq
        )

    # --- Properties to allow external access to weight/bias ---
    @property
    def weight(self):
        return self.original_conv_layer.weight

    @property
    def bias(self):
        return self.original_conv_layer.bias

    def forward(self, x):
        qx = self.activation_quantizer(x)
        qw = self.weight_quantizer(self.original_conv_layer.weight)
        return self.original_conv_layer._conv_forward(qx, qw, self.original_conv_layer.bias)


