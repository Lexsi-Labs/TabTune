from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Sequence
import torch.nn as nn

# This helper function is still useful to validate targets
def _find_linear_module_names(model: nn.Module) -> List[str]:
    """Returns dotted module names for all nn.Linear submodules."""
    return [name for name, module in model.named_modules() if isinstance(module, nn.Linear) and name]

@dataclass(frozen=True)
class LoraTargetConfig:
    target_substrings: Sequence[str]

# This dictionary is the core value of this file
MODEL_LORA_TARGETS: Dict[str, LoraTargetConfig] = {
    "TabPFN": LoraTargetConfig(target_substrings=("encoder.5.layer", "y_encoder.2.layer", "transformer_encoder.layers", "decoder_dict.standard")),
    "TabICL": LoraTargetConfig(target_substrings=("col_embedder.tf_col", "row_interactor", "icl_predictor.tf_icl", "icl_predictor.decoder")),
    "OrionMSP": LoraTargetConfig(target_substrings=("col_embedder.tf_col", "row_interactor", "icl_predictor.tf_icl", "icl_predictor.decoder")),
    "OrionBix": LoraTargetConfig(target_substrings=("col_embedder.tf_col", "row_interactor", "icl_predictor.tf_icl", "icl_predictor.decoder", "biaxial")),
    "TabDPT": LoraTargetConfig(target_substrings=("transformer_encoder", "encoder", "y_encoder", "head")),
    "Mitra": LoraTargetConfig(target_substrings=("x_embedding", "layers", "final_layer")),
    "ConTextTab": LoraTargetConfig(target_substrings=("in_context_encoder", "dense", "output_head", "embeddings")),
}

# This function smartly resolves which layers to target
def resolve_lora_targets(model_name: str, model: nn.Module, override: Sequence[str] | None = None) -> List[str]:
    if override:
        return list(override)
    
    config = MODEL_LORA_TARGETS.get(model_name)
    leaf_names = _find_linear_module_names(model)
    
    if config is None:
        return leaf_names # Default to all linear layers if no config is found
        
    resolved: List[str] = []
    for token in config.target_substrings:
        for leaf in leaf_names:
            if token.lower() in leaf.lower():
                resolved.append(leaf)
    
    # Exclude y_encoder for models where its size is dynamic
    if model_name in ["TabICL", "OrionMSP", "OrionBix"]:
        resolved = [name for name in resolved if "y_encoder" not in name]

    return list(set(resolved)) or leaf_names