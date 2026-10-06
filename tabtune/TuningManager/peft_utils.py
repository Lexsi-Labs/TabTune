from __future__ import annotations

import inspect
import logging
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Type, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


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
    #: Substrings of target leaves whose weight is read as a tensor rather than
    #: applied by calling the module -- typically an attention ``out_proj``
    #: handed to a functional attention call. ``LoRALinear`` adds its delta
    #: inside ``forward``, which such a caller never runs, so those leaves need
    #: :class:`FunctionalWeightLoRALinear` instead or the adapters are a silent
    #: no-op: allocated, reported as trainable, and never affecting an output.
    functional_weight_substrings: Sequence[str] = ()


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
    # NOTE (unchanged behaviour, measured): `col_embedder.tf_col`,
    # `row_interactor` and `icl_predictor.tf_icl` each also match that block's
    # `attn.out_proj`, and model/layers.py passes `self.out_proj.weight` to a
    # functional attention call - so those four wrappers currently have no
    # effect. Perturbing only their adapters moves the output by exactly 0,
    # while perturbing three others moves it by 3.3e-07. Adding
    # `functional_weight_substrings=("out_proj",)` here (and to TabICLv2,
    # OrionMSP, OrionMSPv1.5 and OrionBix, which copy these targets) makes them
    # real. It is left off because turning it on changes the PEFT numerics of
    # five shipped models, which is a decision for whoever owns their tuned
    # hyperparameters - not a side effect of adding a new model.
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
    # Mitra v2 is the same Tab2D architecture as v1 -- the checkpoint differs,
    # the module names do not -- so the targets are identical. The entry exists
    # rather than letting "MitraV2" fall through to `resolve_lora_targets`'
    # "no table -> adapt every linear layer" default, which would quietly give
    # v2 a different (and larger) adapter set than v1 and make the two
    # incomparable in a benchmark.
    "MitraV2": LoraTargetConfig(
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

    # Causilo packs Q/K/V into one `projection` Linear per attention block and
    # keeps `output`; its SwiGLU feedforward holds `value`/`gate`/`output`.
    # Verified against models/causilo/nn/layers/{attention,feedforward}.py.
    #
    # `projection` needs FunctionalWeightLoRALinear. Attention.forward routes
    # through prepare_query/prepare_context, which slice the packed weight -
    # `F.linear(rows, self.projection.weight[: self.width], ...)` - rather than
    # calling the module, and FeatureEmbedding does the same in nn/embeddings.py.
    # Measured on a stub before this was declared: perturbing the adapters on
    # all eight attention/embedding `projection` leaves moved the output by
    # exactly 0, so PEFT was silently adapting only `feature_target.projection`
    # and `row_target.projection`, which are called normally. Slicing the merged
    # weight works and stays differentiable in the adapters.
    "Causilo": LoraTargetConfig(
        target_substrings=(
            "projection",
            "output",
            "value",
            "gate",
        ),
        functional_weight_substrings=("projection",),
    ),

    # Xiaomi TabLDM: the VENDORED tree (tabtune/models/tabldm/_model/).
    # Names below are the model's real nn.Linear leaves: `linear1`/`linear2` are
    # the FFN in the column, row and ICL blocks AND both the routed and shared
    # MoE experts (moe.py::FeedForwardExpert); `attn_res_projs` are the AttnRes
    # residual projections; `in_linear` is the column cell embedder
    # (a SkippableLinear); `ssmax_layer` holds the qassmax attention-scaling
    # MLPs; `decoder` is the ICL head, whose width is the FIXED `max_classes`
    # hyperparam / quantile count, not the dataset's class count, so it is safe
    # to adapt (same reasoning as TabFM).
    #
    # `out_proj` needs FunctionalWeightLoRALinear, hence
    # functional_weight_substrings below: MultiheadAttention.forward hands
    # `self.out_proj.weight` / `.bias` to a functional attention call
    # (_model/layers.py), and a plain LoRALinear - which adds its delta inside
    # forward() - would be a silent no-op there.
    #
    # `in_proj_weight` (the packed QKV) is a raw nn.Parameter, not an nn.Linear
    # submodule, so module-replacement LoRA cannot reach it at all - the same
    # limitation documented for EXAONE below.
    "TabLDM": LoraTargetConfig(
        target_substrings=(
            "linear1",
            "linear2",
            "attn_res_projs",
            "in_linear",
            "ssmax_layer",
            "decoder",
            "out_proj",
        ),
        functional_weight_substrings=("out_proj",),
    ),

    # v3.5 shares the v3 attention module names (q/k/v/out_projection), so the
    # same targets apply; verified against architectures/tabpfn_v3_5.py.
    "TabPFNv35Fast": LoraTargetConfig(
        target_substrings=(
            "q_projection",
            "k_projection",
            "v_projection",
            "out_projection",
            "x_embed",
            "icl_blocks",
        ),
    ),

    "TabPFNv35": LoraTargetConfig(
        target_substrings=(
            "q_projection",
            "k_projection",
            "v_projection",
            "out_projection",
            "x_embed",
            "icl_blocks",
        ),
    ),

    "TabPFNv3": LoraTargetConfig(
        target_substrings=(
            "q_projection",
            "k_projection",
            "v_projection",
            "out_projection",
            "x_embed",
            "icl_blocks",
            "feature_distribution_embedder",
            "column_aggregator",
        ),
    ),
    # TabFM (Google): the VENDORED architecture (tabtune/models/tabfm/model/model.py).
    # Real nn.Linear leaf names -> target the attention projections (q/k/v/out),
    # the swiglu FFN linears (linear1/linear1_gate/linear2), the Fourier cell
    # embedders (in_linear/in_linear_cat), and the column/row/ICL transformer
    # stacks (tf_col/tf_row/tf_icl) + col out_w + the ICL decoder MLP. TabFM's
    # y-encoder/decoder head widths depend on the FIXED model hyperparam
    # `max_classes` (not the dataset's class count), so no exclusions are needed.
    "TabFM": LoraTargetConfig(
        target_substrings=(
            "q_proj",
            "k_proj",
            "v_proj",
            "out_proj",
            "linear1",
            "linear1_gate",
            "linear2",
            "in_linear",
            "in_linear_cat",
            "out_w",
            "tf_col",
            "tf_row",
            "tf_icl",
            "cell_embedder",
            "col_embedder",
            "row_interactor",
            "icl_predictor.decoder",
        ),
    ),
    # iLTM (AI-sandbox): the VENDORED hypernetwork (tabtune/models/iltm/iltm_model.py).
    # ALL trainable nn.Linear leaves live inside the HypernetworkBlock:
    # `hypernetwork_block.hypernetworks.<i>.<j>` (the per-layer hypernetwork MLPs,
    # including the last weight-generating layer) and
    # `hypernetwork_block.hn_emb_to_weights.<i>` (embedding -> main-network-weight
    # projections / optional bottleneck). The generated main network itself is
    # functional (weights are hypernetwork OUTPUTS, not parameters), and the
    # InitialTransformationBlock (random features / PCA / norm) is data-dependent
    # and non-trainable, so there is nothing else to adapt. All widths are fixed
    # model hyperparams (`n_dims`, `hn_hidden_size`, `n_classes_limit`), never the
    # dataset's class count, so no exclusions are needed.
    "ILTM": LoraTargetConfig(
        target_substrings=(
            "hypernetworks",
            "hn_emb_to_weights",
        ),
    ),
    # EXAONE Tabular (LG AI Research): the VENDORED Cross-axis Summary
    # Transformer (tabtune/models/exaone/model/).
    #
    # READ THIS BEFORE TRUSTING THE ENTRY. The names below are the model's real
    # projections, but they are raw `nn.Parameter`s applied through `F.linear`
    # (see model/attention.py: `self.query_weight = nn.Parameter(...)`, then
    # `F.linear(query, self.query_weight)`; and model/mlp.py:
    # `expansion_weight` / `projection_weight`), NOT `nn.Linear` submodules.
    # `inject_custom_lora_into_linear_layers` walks `named_children()` and wraps
    # only `nn.Linear` leaves, so it CANNOT reach any of them. The only
    # `nn.Linear` leaves in the whole model are the two inside
    # `transformer.classification_heads.standard`, and those are excluded below
    # (their output width is the fixed class capacity / quantile count, i.e. the
    # task head, which is the one place LoRA should not be the whole story).
    #
    # The entry is kept -- rather than omitted -- for three reasons: it documents
    # the correct target set for whoever teaches the injector to wrap raw
    # parameters (a `LoRAParameter` shim doing `F.linear(x, W + BA*s)` is the
    # natural fix); it keeps `resolve_lora_targets` from falling through to
    # "adapt every linear layer", which for EXAONE would silently mean "adapt
    # only the task head"; and `TuningManager._warn_if_no_lora_adapters` logs a
    # loud warning when a PEFT run ends up wrapping zero layers, so 'peft' on
    # EXAONE reports honestly that it is currently a full fine-tune.
    "EXAONETabular": LoraTargetConfig(
        target_substrings=(
            "query_weight",
            "key_weight",
            "value_weight",
            "output_weight",
            "expansion_weight",
            "projection_weight",
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


class FunctionalWeightLoRALinear(LoRALinear):
    """LoRA wrapper for leaves read as ``layer.weight`` instead of called.

    ``LoRALinear`` adds its delta in ``forward``. A caller that only takes the
    weight tensor - ``F.linear(x, self.out_proj.weight, self.out_proj.bias)``,
    or a functional attention call - never runs that, so the adapters would
    have no effect at all. Merging the delta into ``weight`` covers both
    consumption styles and stays differentiable in the adapters.

    The merge is recomputed per read, which is an ``r x out x in`` matmul: cheap
    next to the attention it feeds.
    """

    @property
    def weight(self):  # type: ignore[override]
        if self.r <= 0 or self.lora_A is None or self.lora_B is None:
            return self.base.weight
        delta = (self.lora_B.weight @ self.lora_A.weight) * self.scaling
        return self.base.weight + delta.to(self.base.weight.dtype)


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
    functional_weight_patterns: Optional[Sequence[str]] = None,
) -> nn.Module:
    """Inject LoRA adapters into linear layers, optionally excluding certain patterns.

    Leaves matching ``functional_weight_patterns`` get
    :class:`FunctionalWeightLoRALinear` instead of :class:`LoRALinear`, for
    modules whose weight is consumed as a tensor rather than by calling them.
    """
    items = _collect_linear_items(model)
    tokens = [t.lower() for t in target_names or ()]
    exclude_tokens = [e.lower() for e in exclude_patterns or ()]
    functional_tokens = [f.lower() for f in functional_weight_patterns or ()]

    wrapped_count = 0
    for parent, attr, dotted in items:
        # Skip if doesn't match target patterns
        if tokens and not _should_wrap(dotted, tokens):
            continue
        # Skip if matches exclude patterns
        if exclude_tokens and _should_wrap(dotted, exclude_tokens):
            continue
        base_linear = getattr(parent, attr)
        wrapper = (
            FunctionalWeightLoRALinear
            if functional_tokens and _should_wrap(dotted, functional_tokens)
            else LoRALinear
        )
        setattr(parent, attr, wrapper(base_linear, r=r, alpha=alpha, dropout=dropout))
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
        # Resolve through the registry's aliases so a canonical name still finds
        # a table entry written under a different spelling. "ContextTab" missed
        # the "ConTextTab" key here, so it silently fell back to adapting every
        # linear layer in the model instead of its curated target set.
        try:
            from ..registry import get_model_spec

            spec = get_model_spec(model_name)
            for candidate in (spec.name, *spec.aliases):
                config = MODEL_LORA_TARGETS.get(candidate)
                if config is not None:
                    logger.debug(
                        "[PEFT] Resolved LoRA targets for %r via alias %r",
                        model_name,
                        candidate,
                    )
                    break
        except Exception:  # unregistered model, or registry unavailable
            config = None

    if config is None:
        logger.info(
            "[PEFT] No LoRA target table for %r; adapting all linear layers. "
            "Add a MODEL_LORA_TARGETS entry to target specific modules.",
            model_name,
        )
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
    functional_weight_patterns = peft_config.get("functional_weight_modules")
    if functional_weight_patterns is None:
        table_entry = MODEL_LORA_TARGETS.get(model_name)
        functional_weight_patterns = (
            table_entry.functional_weight_substrings if table_entry else ()
        )
    
    # Model-specific exclusions for modules with dynamic dimensions
    exclude_patterns = []
    if model_name in ["TabICL", "OrionMSP", "OrionBix"]:
        exclude_patterns = ["y_encoder"]
    elif model_name == "Causilo":
        # The prediction head's width is the class capacity, and above ten
        # classes Causilo drives it through error-correcting output codes;
        # adapting it would change what those codes decode.
        exclude_patterns = ["head"]
    elif model_name in ("TabPFNv3", "TabPFNv35", "TabPFNv35Fast"):
        # y-encoders can be class-count dependent; keep them full-rank.
        exclude_patterns = ["col_y_encoder", "icl_y_encoder", "y_encoder"]
    # TabFM needs no exclusions: its y-encoder / decoder head widths depend on the
    # fixed `max_classes` model hyperparam (not the dataset class count), so they
    # are safe to adapt with LoRA.
    elif model_name == "TabLDM":
        # `y_encoder` is a OneHotAndLinear whose rows are class slots, and
        # TabLDM permutes class ids per ensemble member
        # (class_shuffle_method="shift"), so those rows are deliberately
        # interchangeable -- an adapter fitted under one permutation does not
        # hold under another. `router` picks which MoE experts fire; it was
        # trained jointly with expert specialisation, so a delta on it re-routes
        # tokens to experts the adapters were never fitted against.
        exclude_patterns = ["y_encoder", "router"]
    elif model_name == "EXAONETabular":
        # Exclude the task head. Its output width is the architectural class
        # capacity (classification) / quantile count (regression), and it is the
        # only nn.Linear pair in the model -- without this exclusion the
        # "no target matched -> adapt every linear layer" fallback in
        # resolve_lora_targets would quietly turn EXAONE PEFT into
        # "LoRA on the output head and nothing else". See the MODEL_LORA_TARGETS
        # comment above.
        exclude_patterns = ["classification_heads"]


    return inject_custom_lora_into_linear_layers(
        model,
        target_names=target_modules,
        r=r,
        alpha=alpha,
        dropout=dropout,
        exclude_patterns=exclude_patterns,
        functional_weight_patterns=functional_weight_patterns,
    )

