"""Minimal LoRA building blocks for the time series adapters.

``LoRALinear`` wraps an ``nn.Linear`` with a frozen base and a trainable
low-rank update ``B A x * alpha / r`` (Hu et al., 2021). It mirrors
``tabtune.TuningManager.peft_utils.LoRALinear`` but imports nothing but torch,
so fine-tuning a forecaster does not load every tabular model.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn

__all__ = ["LoRALinear", "inject_lora", "lora_state_dict", "merge_lora"]


class LoRALinear(nn.Module):
    def __init__(self, base_linear: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        self.base = base_linear
        in_features = base_linear.in_features
        out_features = base_linear.out_features
        self.r = r
        self.scaling = alpha / r if r > 0 else 0.0
        self.lora_A = nn.Linear(in_features, r, bias=False) if r > 0 else None
        self.lora_B = nn.Linear(r, out_features, bias=False) if r > 0 else None
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        if self.lora_A is not None and self.lora_B is not None:
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)
            device = next(base_linear.parameters()).device
            dtype = next(base_linear.parameters()).dtype
            self.lora_A.to(device=device, dtype=dtype)
            self.lora_B.to(device=device, dtype=dtype)
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        base_out = self.base(x)
        if self.r <= 0 or self.lora_A is None or self.lora_B is None:
            return base_out
        x_lora = x.to(dtype=self.lora_A.weight.dtype) if x.dtype != self.lora_A.weight.dtype else x
        lora_out = self.lora_B(self.lora_A(self.dropout(x_lora))) * self.scaling
        return base_out + lora_out

    @property
    def weight(self):  # pragma: no cover - compatibility passthrough
        return self.base.weight

    @property
    def bias(self):  # pragma: no cover - compatibility passthrough
        return self.base.bias


def inject_lora(
    model: nn.Module,
    target_substrings: Sequence[str],
    *,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
) -> list[str]:
    """Wrap every ``nn.Linear`` whose dotted name contains a target substring.

    Returns the dotted names wrapped. The base weights are frozen by
    :class:`LoRALinear`; freezing the rest of the model is the caller's job.
    """
    wrapped: list[str] = []
    tokens = [t.lower() for t in target_substrings]

    def walk(parent: nn.Module, prefix: str) -> None:
        for name, child in list(parent.named_children()):
            dotted = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear) and any(t in dotted.lower() for t in tokens):
                setattr(parent, name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
                wrapped.append(dotted)
            elif not isinstance(child, LoRALinear):
                walk(child, dotted)

    walk(model, "")
    return wrapped


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Only the LoRA A/B tensors of ``model``."""
    return {
        k: v.detach().cpu()
        for k, v in model.state_dict().items()
        if ".lora_A." in k or ".lora_B." in k
    }


def merge_lora(model: nn.Module) -> int:
    """Fold every LoRA update into its base weight and unwrap; returns layers merged."""
    merged = 0

    def walk(parent: nn.Module) -> None:
        nonlocal merged
        for name, child in list(parent.named_children()):
            if isinstance(child, LoRALinear):
                base = child.base
                if child.r > 0 and child.lora_A is not None and child.lora_B is not None:
                    with torch.no_grad():
                        delta = (child.lora_B.weight @ child.lora_A.weight) * child.scaling
                        base.weight += delta.to(base.weight.dtype)
                for p in base.parameters():
                    p.requires_grad = True
                setattr(parent, name, base)
                merged += 1
            else:
                walk(child)

    walk(model)
    return merged
