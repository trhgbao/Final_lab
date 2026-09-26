"""
Weight-Decomposed Low-Rank Adaptation (DoRA) for Video Diffusion Architecture (Pipeline v3).
Based on "DoRA: Weight-Decomposed Low-Rank Adaptation" (Liu et al., ICML 2024 Oral).

Core Formulation:
    W' = m * (V' / ||V'||_c)
    where:
        V' = W0 + (alpha / r) * (B @ A)
        m = learnable magnitude vector initialized as ||W0||_c
        ||.||_c = L2 norm computed along columns/input dimension in float32.

Key Optimization Features (Section 4 & 5 of ICML 2024 paper):
1. Detach Norm in Backpropagation: ||V'||_c is detached when scaling V' to reduce memory
   footprint by ~24.4% without any loss of accuracy.
2. FP32 Stability: Norm is computed in float32 to prevent half-precision (FP16/BF16) overflow/underflow.
3. Parameter Group Isolation: Magnitude vector m must use weight_decay = 0.0 to prevent
   magnitude shrinkage and latent collapse.
"""

import math
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoRALinear(nn.Module):
    """
    DoRA Layer wrapping a base frozen linear layer W0 in CogVideoX Transformer DiT block.
    Decomposes weights into magnitude (m) and direction (V).
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        r: int = 16,
        lora_alpha: float = 32.0,
        lora_dropout: float = 0.0,
    ):
        super().__init__()
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = float(lora_alpha) / float(r)

        # 1. Base weights (100% Frozen)
        self.weight = nn.Parameter(base_linear.weight.data.clone(), requires_grad=False)
        if base_linear.bias is not None:
            self.bias = nn.Parameter(base_linear.bias.data.clone(), requires_grad=False)
        else:
            self.register_parameter("bias", None)

        target_device = base_linear.weight.device
        target_dtype = base_linear.weight.dtype

        # 2. DoRA Low-Rank Direction Matrices: delta_V = (alpha/r) * (B @ A)
        # lora_A initialized with Kaiming uniform, lora_B initialized with 0
        self.lora_A = nn.Parameter(
            torch.zeros((r, self.in_features), device=target_device, dtype=target_dtype)
        )
        self.lora_B = nn.Parameter(
            torch.zeros((self.out_features, r), device=target_device, dtype=target_dtype)
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        # 3. DoRA Magnitude Vector: m in R^(out_features, 1)
        # Initialized to column/row norm of W0
        with torch.no_grad():
            initial_norm = torch.linalg.norm(self.weight.float(), dim=1, keepdim=True)
        self.magnitude = nn.Parameter(
            initial_norm.to(device=target_device, dtype=target_dtype), requires_grad=True
        )
        # Pre-register static base norm buffer to avoid recomputing 180 large matrix norms on every forward step
        self.register_buffer(
            "base_norm",
            torch.clamp(initial_norm, min=1e-8).to(device=target_device, dtype=target_dtype),
            persistent=False,
        )

        # 4. Dropout
        self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()

        # 5. Runtime control flags
        self.lora_scale = 1.0
        self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged:
            return F.linear(x, self.weight, self.bias)

        scale = getattr(self, "lora_scale", 1.0)
        if scale == 0.0:
            return F.linear(x, self.weight, self.bias)

        # Ensure parameter device/dtype alignment
        if self.lora_A.device != self.weight.device or self.lora_A.dtype != self.weight.dtype:
            self.lora_A.data = self.lora_A.data.to(device=self.weight.device, dtype=self.weight.dtype)
            self.lora_B.data = self.lora_B.data.to(device=self.weight.device, dtype=self.weight.dtype)
            self.magnitude.data = self.magnitude.data.to(device=self.weight.device, dtype=self.weight.dtype)

        # 1. Compute directional update: delta_V = B @ A * (scaling * scale)
        lora_drop = self.dropout(self.lora_A)
        delta_V = (self.lora_B @ lora_drop) * (self.scaling * scale)
        V = self.weight + delta_V  # shape: (out_features, in_features)

        # 2. Compute L2 norm across input dimension in FP32 for numerical stability
        # Apply Section 4.3 Detach Norm trick to reduce training memory by 24.4%
        # Use clamp(min=1e-8) instead of +1e-6 to guarantee mathematical equivalence W' == W0 at step 0
        V_norm = torch.clamp(torch.linalg.norm(V.float(), dim=1, keepdim=True), min=1e-8).detach()
        V_norm = V_norm.to(V.dtype)

        # 3. Scale unit direction vector by learnable magnitude m
        # When scale == 1.0 (standard training/inference), directly use self.magnitude without recomputing base_norm
        if scale == 1.0:
            eff_mag = self.magnitude
        else:
            eff_mag = self.base_norm + scale * (self.magnitude - self.base_norm)
        effective_weight = eff_mag * (V / V_norm)
        effective_weight = effective_weight.to(x.dtype)

        return F.linear(x, effective_weight, self.bias)

    def merge_weights(self):
        """Merges DoRA parameters permanently into self.weight for zero-overhead inference."""
        if not self.merged:
            with torch.no_grad():
                scale = getattr(self, "lora_scale", 1.0)
                delta_V = (self.lora_B @ self.lora_A) * (self.scaling * scale)
                V = self.weight + delta_V
                V_norm = torch.clamp(torch.linalg.norm(V.float(), dim=1, keepdim=True), min=1e-8)
                if scale == 1.0:
                    eff_mag = self.magnitude
                else:
                    eff_mag = self.base_norm + scale * (self.magnitude - self.base_norm)
                merged_weight = (eff_mag * (V / V_norm.to(V.dtype))).to(self.weight.dtype)
                self.weight.data.copy_(merged_weight)
                self.merged = True

    def unmerge_weights(self):
        """Unmerges weights to resume training."""
        if self.merged:
            raise RuntimeError("Unmerging is irreversible after in-place weight copy; reload base checkpoint to resume training.")


def apply_dora_to_module(
    module: nn.Module,
    target_keywords: Tuple[str, ...] = (
        "attn1.to_q",
        "attn1.to_k",
        "attn1.to_v",
        "attn1.to_out.0",
        "ff.net.0.proj",
        "ff.net.2",
    ),
    r: int = 16,
    lora_alpha: float = 32.0,
    lora_dropout: float = 0.0,
    prefix: str = "",
) -> Dict[str, DoRALinear]:
    """
    Recursively replaces target linear layers with DoRALinear layers.
    Targets 3D Spatio-Temporal Self-Attention (attn1) and Feed-Forward Network (ff).
    """
    dora_modules = {}
    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear):
            if any(keyword in full_name for keyword in target_keywords):
                dora_layer = DoRALinear(
                    child, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout
                )
                setattr(module, name, dora_layer)
                dora_modules[full_name] = dora_layer
        else:
            child_dora = apply_dora_to_module(
                child,
                target_keywords=target_keywords,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                prefix=full_name,
            )
            dora_modules.update(child_dora)
    return dora_modules


def get_dora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Extracts all DoRA parameters (lora_A, lora_B, magnitude) from the model regardless of requires_grad state."""
    dora_state_dict = {}
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name or "magnitude" in name:
            dora_state_dict[name] = param.detach().cpu()
    return dora_state_dict


def load_dora_state_dict(model: nn.Module, state_dict: Dict[str, torch.Tensor], strict: bool = False):
    """Loads DoRA parameters into the model with robust dtype and device casting."""
    params_dict = dict(model.named_parameters())
    for name, tensor in state_dict.items():
        if name in params_dict:
            params_dict[name].data.copy_(tensor.to(device=params_dict[name].device, dtype=params_dict[name].dtype))
        elif strict:
            raise KeyError(f"Unexpected DoRA parameter: {name}")


def get_dora_param_groups(
    model: nn.Module,
    lr_direction: float = 2e-4,
    lr_magnitude: float = 1e-4,
    weight_decay_direction: float = 1e-2,
) -> List[Dict]:
    """
    Creates isolated optimizer parameter groups for DoRA training according to ICML 2024 paper:
    1. Direction matrices (lora_A, lora_B): Standard learning rate, standard weight decay.
    2. Magnitude vectors (magnitude): Learning rate, CRITICAL: weight_decay = 0.0 (prevents norm shrinkage).
    """
    direction_params = []
    magnitude_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "magnitude" in name:
            magnitude_params.append(param)
        elif "lora_A" in name or "lora_B" in name:
            direction_params.append(param)

    param_groups = [
        {
            "params": direction_params,
            "lr": lr_direction,
            "weight_decay": weight_decay_direction,
            "name": "dora_direction",
        },
        {
            "params": magnitude_params,
            "lr": lr_magnitude,
            "weight_decay": 0.0,  # CRITICAL: Absolutely no weight decay on magnitude!
            "name": "dora_magnitude",
        },
    ]
    return param_groups
