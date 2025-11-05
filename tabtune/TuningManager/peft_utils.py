from __future__ import annotations

import inspect
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Type, Union

import torch
import torch.nn as nn


def _find_linear_module_names(model: torch.nn.Module) -> List[str]:
    """
    Returns dotted module names for all nn.Linear submodules.
    """
    linear_names: List[str] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name:
            linear_names.append(name)
    return linear_names


def _collect_linear_items(model: torch.nn.Module) -> List[Tuple[nn.Module, str, str]]:
    """
    Returns list of (parent_module, attr_name, dotted_name) for every nn.Linear leaf.
    """
    items: List[Tuple[nn.Module, str, str]] = []

    def _walk(parent: nn.Module, prefix: str = "") -> None:
        for child_name, child in parent.named_children():
            dotted = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, nn.Linear):
                items.append((parent, child_name, dotted))
            else:
                _walk(child, dotted)

    _walk(model)
    return items


@dataclass(frozen=True)
class LoraTargetConfig:
    target_substrings: Sequence[str]
    task_type: str = "FEATURE_EXTRACTION"


# --- Per-model defaults -----------------------------------------------------
MODEL_LORA_TARGETS: Dict[str, LoraTargetConfig] = {
    "TabPFN": LoraTargetConfig(
        target_substrings=(
            "encoder.5.layer",
            "y_encoder.2.layer",
            "transformer_encoder.layers",
            "decoder_dict.standard.0",
            "decoder_dict.standard.2",
            "feature_positional_embedding_embeddings",
        ),
    ),
    "TabICL": LoraTargetConfig(
        target_substrings=(
            "col_embedder.tf_col",
            "col_embedder.in_linear",
            "col_embedder.out_w",
            "col_embedder.out_b",
            "row_interactor",
            "icl_predictor.tf_icl",
            "icl_predictor.decoder",
            # Exclude y_encoder as it has dynamic dimensions based on num_classes
        ),
    ),
    "OrionMSP": LoraTargetConfig(
        target_substrings=(
            "col_embedder.tf_col",
            "col_embedder.in_linear",
            "col_embedder.out_w",
            "col_embedder.out_b",
            "row_interactor",
            "icl_predictor.tf_icl",
            "icl_predictor.decoder",
            # Exclude y_encoder as it has dynamic dimensions based on num_classes
        ),
    ),
    "OrionBix": LoraTargetConfig(
        target_substrings=(
            "col_embedder.tf_col",
            "col_embedder.in_linear",
            "col_embedder.out_w",
            "col_embedder.out_b",
            "row_interactor",
            "icl_predictor.tf_icl",
            "icl_predictor.decoder",
            "biaxial",
            # Exclude y_encoder as it has dynamic dimensions based on num_classes
        ),
    ),
    "TabDPT": LoraTargetConfig(
        target_substrings=(
            "transformer_encoder",
            "encoder",
            "y_encoder",
            "head",
        ),
    ),
    "Mitra": LoraTargetConfig(
        target_substrings=(
            "x_embedding",
            "layers",
            "final_layer",
        ),
    ),
    "ConTextTab": LoraTargetConfig(
        target_substrings=(
            "in_context_encoder",
            "dense",
            "output_head",
            "embeddings",
        ),
    ),
}


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
            # Move LoRA adapters to same device and dtype as base layer
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
        # Cast input to match LoRA adapter dtype if needed
        x_lora = x.to(dtype=self.lora_A.weight.dtype) if x.dtype != self.lora_A.weight.dtype else x
        lora_out = self.lora_B(self.lora_A(self.dropout(x_lora))) * self.scaling
        return base_out + lora_out

    @property
    def weight(self):  # pragma: no cover - compatibility passthrough
        return self.base.weight

    @property
    def bias(self):  # pragma: no cover - compatibility passthrough
        return self.base.bias


def _should_wrap(name: str, targets: Sequence[str]) -> bool:
    lowered = name.lower()
    return any(tok.lower() in lowered for tok in targets)


def inject_custom_lora_into_linear_layers(
    model: nn.Module,
    target_names: Optional[Sequence[str]] = None,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
    exclude_patterns: Optional[Sequence[str]] = None,
) -> nn.Module:
    """Inject LoRA adapters into linear layers, optionally excluding certain patterns."""
    items = _collect_linear_items(model)
    tokens = [t.lower() for t in target_names or ()]
    exclude_tokens = [e.lower() for e in exclude_patterns or ()]

    wrapped_count = 0
    for parent, attr, dotted in items:
        # Skip if doesn't match target patterns
        if tokens and not _should_wrap(dotted, tokens):
            continue
        # Skip if matches exclude patterns
        if exclude_tokens and _should_wrap(dotted, exclude_tokens):
            continue
        base_linear = getattr(parent, attr)
        setattr(parent, attr, LoRALinear(base_linear, r=r, alpha=alpha, dropout=dropout))
        wrapped_count += 1

    return model


def resolve_lora_targets(
    model_name: str,
    model: nn.Module,
    override: Optional[Sequence[str]] = None,
) -> Sequence[str]:
    if override:
        return override
    config = MODEL_LORA_TARGETS.get(model_name)
    if config is None:
        return _find_linear_module_names(model)
    # Only keep leaves that actually exist
    leaf_names = _find_linear_module_names(model)
    resolved: List[str] = []
    for token in config.target_substrings:
        for leaf in leaf_names:
            if token.lower() in leaf.lower():
                resolved.append(leaf)
    return resolved or leaf_names


def apply_tabular_lora(
    model_name: str,
    model: nn.Module,
    peft_config: Optional[Dict] = None,
) -> nn.Module:
    if peft_config is None:
        peft_config = {}
    r = peft_config.get("r", 8)
    alpha = peft_config.get("lora_alpha", 16)
    dropout = peft_config.get("lora_dropout", 0.05)
    target_modules = resolve_lora_targets(model_name, model, peft_config.get("target_modules"))
    
    # Model-specific exclusions for modules with dynamic dimensions
    exclude_patterns = []
    if model_name in ["TabICL", "OrionMSP", "OrionBix"]:
        exclude_patterns = ["y_encoder"]
    
    return inject_custom_lora_into_linear_layers(
        model,
        target_names=target_modules,
        r=r,
        alpha=alpha,
        dropout=dropout,
        exclude_patterns=exclude_patterns,
    )

