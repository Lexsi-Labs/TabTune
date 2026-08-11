"""Cross-Axis Summary Transformer (CAST) stack used by the classification model."""

from __future__ import annotations

import math
import numbers
from collections.abc import Callable
from typing import TypeVar

import torch
from torch import nn

from ..config import ModelConfig
from .layer import CrossAxisSummaryLayer
from ..runtime import FULL_FEATURE_ATTENTION_ROWS


_T = TypeVar("_T", bound=torch.Tensor)
_FLOAT_DTYPES = {torch.float16, torch.bfloat16, torch.float32, torch.float64}


def run_in_chunks(
    function: Callable[[torch.Tensor], _T],
    value: torch.Tensor,
    chunk_size: int,
    *,
    add_input: bool = False,
) -> _T:
    """Evaluate ``function`` over consecutive leading-axis slices."""

    # An empty input still has to pass through the module once.  Besides being
    # useful for hooks, this lets Linear determine the correctly shaped result.
    if value.shape[0] == 0:
        result = function(value)
        return result + value if add_input else result

    pieces = []
    for start in range(0, value.shape[0], chunk_size):
        source = value[start : start + chunk_size]
        result = function(source)
        pieces.append(result + source if add_input else result)
    return torch.cat(pieces, dim=0)  # type: ignore[return-value]


class CrossAxisSummaryTransformer(nn.Module):
    """Apply cross-axis summary layers and classify the query-row readout tokens."""

    def __init__(
        self,
        config: ModelConfig,
        *,
        activation: str = "gelu",
        layer_norm_epsilon: float = 1e-5,
        initial_head_scale: float = 0.43,
        learned_head_scale: bool = True,
        score_scale: float | None = None,
        output_width: int | None = None,
        decoder_hidden_width: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        zero_residual_branches: bool = False,
    ) -> None:
        super().__init__()
        self._validate_constructor(
            config,
            activation,
            layer_norm_epsilon,
            initial_head_scale,
            learned_head_scale,
            score_scale,
            output_width,
            decoder_hidden_width,
            dtype,
            zero_residual_branches,
        )

        factory = {"device": device, "dtype": dtype}
        self.layers = nn.ModuleList(
            CrossAxisSummaryLayer(
                config,
                config.feedforward_widths[index],
                activation=activation,
                layer_norm_epsilon=float(layer_norm_epsilon),
                initial_head_scale=float(initial_head_scale),
                learned_head_scale=learned_head_scale,
                score_scale=None if score_scale is None else float(score_scale),
                final_layer=index == config.block_count - 1,
                device=device,
                dtype=dtype,
                zero_residual_branches=zero_residual_branches,
            )
            for index in range(config.block_count)
        )

        self.feature_summary_tokens = nn.Parameter(
            torch.empty(1, config.feature_summary_count, 1, config.width, **factory)
        )
        self.item_summary_tokens = nn.Parameter(
            torch.empty(1, 1, config.item_summary_count, config.width, **factory)
        )
        nn.init.normal_(self.feature_summary_tokens, std=float(initial_head_scale))
        nn.init.normal_(self.item_summary_tokens, std=float(initial_head_scale))

        hidden_width = (
            config.feedforward_widths[-1]
            if decoder_hidden_width is None
            else decoder_hidden_width
        )
        resolved_output_width = (
            config.class_capacity if output_width is None else output_width
        )
        self.classification_heads = nn.ModuleDict(
            {
                "standard": nn.Sequential(
                    nn.Linear(config.width, hidden_width, **factory),
                    nn.GELU(),
                    nn.Linear(hidden_width, resolved_output_width, **factory),
                )
            }
        )
        self._width = config.width
        self._class_capacity = resolved_output_width

    @staticmethod
    def _validate_constructor(
        config: ModelConfig,
        activation: object,
        layer_norm_epsilon: object,
        initial_head_scale: object,
        learned_head_scale: object,
        score_scale: object,
        output_width: object,
        decoder_hidden_width: object,
        dtype: object,
        zero_residual_branches: object,
    ) -> None:
        if not isinstance(config, ModelConfig):
            raise TypeError("config must be a ModelConfig")
        if not isinstance(activation, str):
            raise TypeError("activation must be a string")
        if activation not in {"gelu", "relu"}:
            raise ValueError("unsupported activation")

        def real(name: str, value: object, *, positive: bool) -> None:
            if isinstance(value, bool) or not isinstance(value, numbers.Real):
                raise TypeError(f"{name} must be a real scalar")
            number = float(value)
            if not math.isfinite(number) or (positive and number <= 0):
                raise ValueError(f"invalid {name}")

        real("layer_norm_epsilon", layer_norm_epsilon, positive=True)
        real("initial_head_scale", initial_head_scale, positive=False)
        if not isinstance(learned_head_scale, bool):
            raise TypeError("learned_head_scale must be bool")
        if score_scale is not None:
            real("score_scale", score_scale, positive=True)
        for name, value in (
            ("output_width", output_width),
            ("decoder_hidden_width", decoder_hidden_width),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int)
            ):
                raise TypeError(f"{name} must be an integer or None")
            if isinstance(value, int) and value <= 0:
                raise ValueError(f"{name} must be positive")
        if not isinstance(zero_residual_branches, bool):
            raise TypeError("zero_residual_branches must be bool")
        if dtype is not None and not isinstance(dtype, torch.dtype):
            raise TypeError("dtype must be a torch.dtype or None")
        if dtype is not None and dtype not in _FLOAT_DTYPES:
            raise ValueError("unsupported dtype")

    @property
    def has_cached_support(self) -> bool:
        return bool(self.layers) and all(
            bool(getattr(layer, "has_cached_support", False)) for layer in self.layers
        )

    def clear_cache(self) -> None:
        for layer in self.layers:
            layer.clear_cache()

    def load_cached_context(self, device: torch.device | str) -> None:
        """Bulk-reload every layer's offloaded support K/V onto ``device`` once.

        Paired with a ``cache_support_to_cpu`` build: the build keeps the GPU
        working region clear by offloading each layer's K/V to the host as it is
        produced; this restores them all before the query phase so cached
        cross-attention reads stay on the accelerator. No-op on a CPU model.
        """
        for layer in self.layers:
            layer.reload_support_cache(device)

    def _validate_forward(
        self,
        state: object,
        support_size: object,
        output: object,
        cache_support: object,
        cache_support_to_cpu: object,
        use_cached_support: object,
        compact_cached_support: object,
        query_chunk_size: object,
        feedforward_token_chunk: object,
        feature_attention_row_chunk: object,
    ) -> torch.Tensor:
        if not isinstance(state, torch.Tensor):
            raise TypeError("state must be a torch.Tensor")
        if state.dtype not in _FLOAT_DTYPES:
            raise TypeError("state must have a supported floating dtype")
        if state.ndim != 4:
            raise ValueError("state must have rank four")
        if any(size <= 0 for size in state.shape[:3]) or state.shape[3] != self._width:
            raise ValueError("state has an invalid shape")
        if (
            not isinstance(cache_support, bool)
            or not isinstance(cache_support_to_cpu, bool)
            or not isinstance(use_cached_support, bool)
            or not isinstance(compact_cached_support, bool)
        ):
            raise TypeError("cache options must be bool")
        if not isinstance(output, str):
            raise TypeError("output must be a string")
        if output not in {"standard", "all"}:
            raise ValueError("unsupported output mode")
        if query_chunk_size is not None:
            if isinstance(query_chunk_size, bool) or not isinstance(query_chunk_size, int):
                raise TypeError("query_chunk_size must be an integer or None")
            if query_chunk_size <= 0:
                raise ValueError("query_chunk_size must be positive")
        if (
            isinstance(feedforward_token_chunk, bool)
            or not isinstance(feedforward_token_chunk, int)
        ):
            raise TypeError("feedforward_token_chunk must be an integer")
        if feedforward_token_chunk <= 0:
            raise ValueError("feedforward_token_chunk must be positive")
        if (
            isinstance(feature_attention_row_chunk, bool)
            or not isinstance(feature_attention_row_chunk, int)
        ):
            raise TypeError("feature_attention_row_chunk must be an integer")
        if feature_attention_row_chunk <= 0:
            raise ValueError("feature_attention_row_chunk must be positive")
        if isinstance(support_size, bool) or not isinstance(support_size, int):
            raise ValueError("support_size must be an integer")
        if cache_support and use_cached_support:
            raise ValueError("cache modes are mutually exclusive")
        if cache_support_to_cpu and not cache_support:
            raise ValueError("CPU offload requires cache_support")
        if compact_cached_support and not use_cached_support:
            raise ValueError("compact cache execution requires use_cached_support")
        rows = state.shape[1]
        if use_cached_support:
            if support_size < 1 or support_size >= rows:
                raise ValueError("cached-support calls require support and query rows")
            if not self.has_cached_support:
                raise RuntimeError("no complete support cache is available")
        elif support_size < 1 or support_size > rows:
            raise ValueError("support_size is outside the row range")
        if cache_support and support_size <= 0:
            raise ValueError("cache writes require support rows")

        parameter = next(self.parameters())
        if state.device != parameter.device or state.dtype != parameter.dtype:
            raise ValueError("state placement must match module parameters")
        return state

    def forward(
        self,
        state: torch.Tensor,
        support_size: int,
        *,
        feedforward_token_chunk: int,
        feature_attention_row_chunk: int = FULL_FEATURE_ATTENTION_ROWS,
        output: str = "standard",
        cache_support: bool = False,
        cache_support_to_cpu: bool = False,
        use_cached_support: bool = False,
        compact_cached_support: bool = False,
        query_chunk_size: int | None = None,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        state = self._validate_forward(
            state,
            support_size,
            output,
            cache_support,
            cache_support_to_cpu,
            use_cached_support,
            compact_cached_support,
            query_chunk_size,
            feedforward_token_chunk,
            feature_attention_row_chunk,
        )

        if compact_cached_support:
            state = state[:, support_size:]

        batch, rows, feature_groups, _ = state.shape
        feature_summaries = self.feature_summary_tokens.expand(
            batch, -1, feature_groups, -1
        )
        item_summaries = self.item_summary_tokens.expand(batch, rows, -1, -1)

        if cache_support:
            # Cache replacement is transactional at the stack level.
            self.clear_cache()
        try:
            for layer in self.layers:
                state, feature_summaries, item_summaries = layer(
                    state,
                    feature_summaries,
                    item_summaries,
                    support_size,
                    feedforward_token_chunk=feedforward_token_chunk,
                    feature_attention_row_chunk=feature_attention_row_chunk,
                    cache_support=cache_support,
                    cache_support_to_cpu=cache_support_to_cpu,
                    use_cached_support=use_cached_support,
                    compact_cached_support=compact_cached_support,
                    output_support=output == "all" or not layer.final_layer,
                )
        except Exception:
            if cache_support:
                self.clear_cache()
            raise

        embeddings = state[:, :, -1, :]
        if compact_cached_support:
            support_embeddings = embeddings[:, :0]
            query_embeddings = embeddings
        elif use_cached_support:
            support_embeddings = embeddings[:, :support_size]
            query_embeddings = embeddings[:, support_size:]
        else:
            support_embeddings = embeddings[:, :support_size]
            query_embeddings = embeddings[:, support_size:]

        head = self.classification_heads["standard"]
        if query_chunk_size is None:
            logits = head(query_embeddings)
        else:
            flat = query_embeddings.reshape(-1, self._width)
            flat_logits = run_in_chunks(
                head, flat, query_chunk_size, add_input=False
            )
            logits = flat_logits.reshape(
                batch, query_embeddings.shape[1], self._class_capacity
            )

        if output == "standard":
            return logits
        return {
            "standard": logits,
            "support_embeddings": support_embeddings,
            "query_embeddings": query_embeddings,
        }


__all__ = ["CrossAxisSummaryTransformer"]
