"""Cross-Axis Summary Transformer (CAST) layer for classification models."""

from __future__ import annotations

import functools
import inspect
import math
import numbers
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from ..config import ModelConfig
from .attention import TensorAttention
from .mlp import FeedForward
from ..runtime import FULL_FEATURE_ATTENTION_ROWS, run_in_chunks


_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)


def _config_value(config: ModelConfig, *names: str, default: Any = None) -> Any:
    for name in names:
        if hasattr(config, name):
            return getattr(config, name)
    return default


def _construct(component: type[nn.Module], values: dict[str, Any]) -> nn.Module:
    """Construct a frozen public primitive without coupling to spelling aliases."""
    signature = inspect.signature(component.__init__)
    kwargs: dict[str, Any] = {}
    missing: list[str] = []
    for name, parameter in signature.parameters.items():
        if name == "self" or parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        if name in values and values[name] is not None:
            kwargs[name] = values[name]
        elif parameter.default is inspect.Parameter.empty:
            missing.append(name)
    if missing:
        raise TypeError(
            f"Cannot construct {component.__name__}; unsupported required arguments: "
            + ", ".join(missing)
        )
    return component(**kwargs)


class CrossAxisSummaryLayer(nn.Module):
    """Mix feature and item summary tokens with row-by-feature state."""

    def __init__(
        self,
        config: ModelConfig,
        expanded_width: int,
        *,
        activation: str = "gelu",
        layer_norm_epsilon: float = 1e-5,
        final_layer: bool = False,
        initial_head_scale: float = 0.43,
        learned_head_scale: bool = True,
        score_scale: float | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        zero_residual_branches: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(config, ModelConfig):
            raise TypeError("config must be a ModelConfig")
        if isinstance(expanded_width, bool) or not isinstance(expanded_width, int):
            raise TypeError("expanded_width must be an integer")
        if expanded_width <= 0:
            raise ValueError("expanded_width must be positive")
        if not isinstance(activation, str):
            raise TypeError("activation must be a string")
        if activation not in ("gelu", "relu"):
            raise ValueError("activation must be 'gelu' or 'relu'")
        for name, value, positive in (
            ("layer_norm_epsilon", layer_norm_epsilon, True),
            ("initial_head_scale", initial_head_scale, False),
            ("score_scale", score_scale, True),
        ):
            if value is None and name == "score_scale":
                continue
            if isinstance(value, bool) or not isinstance(value, numbers.Real):
                raise TypeError(f"{name} must be a real scalar")
            if not math.isfinite(float(value)) or (positive and value <= 0):
                raise ValueError(f"invalid {name}")
        if not isinstance(final_layer, bool):
            raise TypeError("final_layer must be bool")
        if not isinstance(learned_head_scale, bool):
            raise TypeError("learned_head_scale must be bool")
        if not isinstance(zero_residual_branches, bool):
            raise TypeError("zero_residual_branches must be bool")
        if dtype is not None and not isinstance(dtype, torch.dtype):
            raise TypeError("dtype must be a torch.dtype or None")
        if dtype is not None and dtype not in _FLOAT_DTYPES:
            raise ValueError("unsupported dtype")

        self.width = int(config.width)
        self.feature_summary_count = int(config.feature_summary_count)
        self.item_summary_count = int(config.item_summary_count)
        self.feature_attention_repeats = int(config.feature_attention_repeats)
        self.layer_norm_epsilon = float(layer_norm_epsilon)
        self.final_layer = final_layer

        common = {
            "width": self.width,
            "embedding_width": self.width,
            "embedding_dim": self.width,
            "model_width": self.width,
            "dim": self.width,
            "initial_head_scale": initial_head_scale,
            "head_scale": initial_head_scale,
            "learned_head_scale": learned_head_scale,
            "score_scale": score_scale,
            # Absent from the frozen ModelConfig, so the per-call policy applies
            # unless a caller's config pins one kernel for the whole model.
            "sdpa_backend": _config_value(config, "sdpa_backend"),
            "device": device,
            "dtype": dtype,
        }
        feature_heads = _config_value(
            config,
            "feature_attention_heads",
            "feature_head_count",
            "feature_heads",
            "num_feature_heads",
            "attention_heads",
            "head_count",
            "num_heads",
        )
        item_heads = _config_value(
            config,
            "item_attention_heads",
            "item_head_count",
            "item_heads",
            "num_item_heads",
            "attention_heads",
            "head_count",
            "num_heads",
            default=feature_heads,
        )
        item_kv_heads = _config_value(
            config,
            "item_attention_kv_heads",
            "item_kv_head_count",
            "item_kv_heads",
            "num_item_kv_heads",
            default=item_heads,
        )

        def attention(
            heads: Any, kv_heads: Any = None, *, sdpa_backend: Any = None
        ) -> TensorAttention:
            values = dict(common)
            if sdpa_backend is not None:
                values["sdpa_backend"] = sdpa_backend
            values.update(
                {
                    "heads": heads,
                    "num_heads": heads,
                    "head_count": heads,
                    "query_heads": heads,
                    "kv_heads": kv_heads,
                    "num_kv_heads": kv_heads,
                    "kv_head_count": kv_heads,
                    "key_value_heads": kv_heads,
                }
            )
            return _construct(TensorAttention, values)  # type: ignore[return-value]

        self.feature_attentions = nn.ModuleList(
            attention(feature_heads) for _ in range(self.feature_attention_repeats)
        )
        # Rows are this call's sequence axis *and* the axis row-chunking splits,
        # so a per-call kernel choice here would follow the memory budget.
        self.feature_context_attentions = nn.ModuleList(
            attention(feature_heads, sdpa_backend=common["sdpa_backend"] or "flash")
            for _ in range(
                self.feature_attention_repeats
                if self.feature_summary_count > 1
                else 0
            )
        )
        self.item_attention = attention(item_heads, item_kv_heads)
        self.item_context_attention = (
            attention(item_heads) if self.item_summary_count > 1 else None
        )

        ff_values = {
            "width": self.width,
            "embedding_width": self.width,
            "embedding_dim": self.width,
            "model_width": self.width,
            "input_width": self.width,
            "dim": self.width,
            "expanded_width": expanded_width,
            "hidden_width": expanded_width,
            "intermediate_width": expanded_width,
            "hidden_dim": expanded_width,
            "activation": activation,
            "device": device,
            "dtype": dtype,
        }

        def feedforward() -> FeedForward:
            return _construct(FeedForward, ff_values)  # type: ignore[return-value]

        self.state_feedforward = feedforward()
        self.feature_summary_feedforward = None if final_layer else feedforward()
        self.item_summary_feedforward = None if final_layer else feedforward()

        self._cached_feature_input: torch.Tensor | None = None
        self._cached_summary_rows: torch.Tensor | None = None
        self._cached_support_rows: int | None = None

        if zero_residual_branches:
            with torch.no_grad():
                for module in self.modules():
                    if isinstance(module, TensorAttention):
                        module.output_weight.zero_()
                    elif isinstance(module, FeedForward):
                        module.projection_weight.zero_()

    @property
    def has_cached_support(self) -> bool:
        return (
            self._cached_feature_input is not None
            and self._cached_summary_rows is not None
            and self._cached_support_rows is not None
            and self.item_attention.has_cached_context
        )

    def clear_cache(self) -> None:
        self._cached_feature_input = None
        self._cached_summary_rows = None
        self._cached_support_rows = None
        self.item_attention.clear_cache()

    def reload_support_cache(self, device: torch.device | str) -> None:
        """Restore this layer's offloaded item-attention K/V onto ``device``."""
        self.item_attention.reload_cached_context(device)

    @staticmethod
    def _attention(
        module: TensorAttention,
        query: torch.Tensor,
        context: torch.Tensor | None = None,
        *,
        shared_context_head: bool = False,
        cache_context: bool = False,
        use_cached_context: bool = False,
    ) -> torch.Tensor:
        return module(
            query,
            context,
            cache_context=cache_context,
            use_cached_context=use_cached_context,
            reuse_first_context_head=shared_context_head,
        )

    @staticmethod
    def _feedforward(
        module: FeedForward,
        value: torch.Tensor,
        token_chunk: int,
    ) -> torch.Tensor:
        flat = value.reshape(-1, value.shape[-1])

        def apply(chunk: torch.Tensor) -> torch.Tensor:
            return module(chunk, add_residual=True)

        if flat.shape[0] <= token_chunk:
            return apply(flat).reshape_as(value)
        chunks = [
            apply(flat[start : start + token_chunk])
            for start in range(0, flat.shape[0], token_chunk)
        ]
        return torch.cat(chunks, dim=0).reshape_as(value)

    def _normalize(self, value: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(value, (self.width,), eps=self.layer_norm_epsilon)

    def _validate_inputs(
        self,
        state: torch.Tensor,
        feature_summary: torch.Tensor,
        item_summary: torch.Tensor,
        support_size: int,
        cache_support: bool,
        cache_support_to_cpu: bool,
        use_cached_support: bool,
        compact_cached_support: bool,
    ) -> None:
        for name, value in (
            ("state", state),
            ("feature_summary", feature_summary),
            ("item_summary", item_summary),
        ):
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a tensor")
            if value.dtype not in _FLOAT_DTYPES:
                raise TypeError(f"{name} has an unsupported dtype")
            if value.ndim != 4:
                raise ValueError(f"{name} must have rank four")
            if any(size <= 0 for size in value.shape):
                raise ValueError(f"{name} dimensions must be positive")
        batch, rows, groups, width = state.shape
        if width != self.width:
            raise ValueError("state width does not match config")
        if feature_summary.shape != (
            batch,
            self.feature_summary_count,
            groups,
            width,
        ):
            raise ValueError("feature_summary shape is incompatible")
        if item_summary.shape != (batch, rows, self.item_summary_count, width):
            raise ValueError("item_summary shape is incompatible")
        if (
            not isinstance(cache_support, bool)
            or not isinstance(cache_support_to_cpu, bool)
            or not isinstance(use_cached_support, bool)
            or not isinstance(compact_cached_support, bool)
        ):
            raise TypeError("cache options must be bool")
        if cache_support and use_cached_support:
            raise ValueError("cache options are mutually exclusive")
        if cache_support_to_cpu and not cache_support:
            raise ValueError("CPU offload requires cache_support")
        if compact_cached_support and not use_cached_support:
            raise ValueError("compact cache execution requires use_cached_support")
        if isinstance(support_size, bool) or not isinstance(support_size, int):
            raise ValueError("support_size must be an integer")
        if use_cached_support:
            if (
                self._cached_support_rows is None
                or support_size != self._cached_support_rows
                or (not compact_cached_support and support_size >= rows)
            ):
                raise ValueError(
                    "cached calls require the fitted support prefix and query rows"
                )
        elif support_size < 1 or support_size > rows:
            raise ValueError("support_size is outside the row range")
        if cache_support and support_size <= 0:
            raise ValueError("cache writes require positive support_size")

        if state.dtype != feature_summary.dtype or state.dtype != item_summary.dtype:
            raise ValueError("input dtypes differ")
        if state.device != feature_summary.device or state.device != item_summary.device:
            raise ValueError("input devices differ")
        parameter = next(self.parameters())
        if state.dtype != parameter.dtype or state.device != parameter.device:
            raise ValueError("inputs do not match module parameter placement")

    def _feature_attention_block(
        self,
        state_block: torch.Tensor,
        item_summary_block: torch.Tensor,
        *,
        feature_summary: torch.Tensor,
        attention: TensorAttention,
        repeat: int,
        label_group_only: bool,
    ) -> torch.Tensor:
        """Run one feature-attention repeat over a block of rows.

        Rows are a batch axis in feature attention, so a block is processed
        independently of every other row while reading the shared, read-only
        ``feature_summary``. The returned token block is ``item_summary``
        concatenated with ``state`` along dim 2, already normalized -- LayerNorm
        is token-wise over ``width``, so normalizing the joined block equals
        normalizing the two halves separately. This is what makes the chunked
        and full-row paths agree.
        """
        if self.feature_summary_count == 1:
            feature_context = state_block + feature_summary[:, 0].unsqueeze(1)
        else:
            query = state_block.transpose(1, 2)
            context = feature_summary.transpose(1, 2)
            mixed = self._attention(
                self.feature_context_attentions[repeat], query, context
            ).transpose(1, 2)
            feature_context = state_block + mixed
        joined = torch.cat((item_summary_block, feature_context), dim=2)
        if label_group_only:
            # Standard output consumes only the final label group. The last
            # feature-attention repeat still attends to the complete token
            # context, but feature-group queries have no later consumer in the
            # final transformer layer.
            selected = torch.cat(
                (joined[:, :, : self.item_summary_count], joined[:, :, -1:]),
                dim=2,
            )
            selected = selected + self._attention(attention, selected, joined)
            return self._normalize(selected)
        joined = joined + self._attention(attention, joined)
        return self._normalize(joined)

    def forward(
        self,
        state: torch.Tensor,
        feature_summary: torch.Tensor,
        item_summary: torch.Tensor,
        support_size: int,
        *,
        feedforward_token_chunk: int,
        feature_attention_row_chunk: int = FULL_FEATURE_ATTENTION_ROWS,
        cache_support: bool = False,
        cache_support_to_cpu: bool = False,
        use_cached_support: bool = False,
        compact_cached_support: bool = False,
        output_support: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        if not isinstance(output_support, bool):
            raise TypeError("output_support must be bool")
        if not output_support and not self.final_layer:
            raise ValueError("support output may only be omitted in the final layer")
        self._validate_inputs(
            state,
            feature_summary,
            item_summary,
            support_size,
            cache_support,
            cache_support_to_cpu,
            use_cached_support,
            compact_cached_support,
        )
        feature_input = feature_summary
        cache_summary_rows: torch.Tensor | None = None
        label_group_only = False
        if use_cached_support:
            if not self.has_cached_support:
                raise RuntimeError("support cache is unavailable")
            assert self._cached_feature_input is not None
            assert self._cached_summary_rows is not None
            cached = self._cached_summary_rows
            if (
                cached.shape[0] != state.shape[0]
                or cached.shape[2] != state.shape[2]
                or cached.dtype != state.dtype
                or cached.device != state.device
                or not torch.equal(feature_summary, self._cached_feature_input)
            ):
                raise RuntimeError("inputs are incompatible with the support cache")

        for repeat, attention in enumerate(self.feature_attentions):
            # Row-independent, so decided once and shared by every row block.
            label_group_only = (
                not output_support
                and repeat == len(self.feature_attentions) - 1
                and "forward" not in attention.__dict__
            )
            block = functools.partial(
                self._feature_attention_block,
                feature_summary=feature_summary,
                attention=attention,
                repeat=repeat,
                label_group_only=label_group_only,
            )
            rows = state.shape[1]
            if feature_attention_row_chunk >= rows:
                combined = block(state, item_summary)
            else:
                combined = run_in_chunks(
                    block,
                    state,
                    item_summary,
                    chunk_size=feature_attention_row_chunk,
                    axis=1,
                )
            item_summary = combined[:, :, : self.item_summary_count]
            state = combined[:, :, self.item_summary_count :]
            feature_summary = self._normalize(feature_summary)
            if label_group_only:
                # Keep a full-group shape only for cache compatibility checks;
                # every downstream computation and the classifier need the
                # appended label group alone.
                cache_summary_rows = feature_summary
                feature_summary = feature_summary[:, :, -1:, :]

        if self.item_summary_count == 1:
            item_context = state + item_summary[:, :, 0].unsqueeze(2)
        else:
            assert self.item_context_attention is not None
            item_context = state + self._attention(
                self.item_context_attention, state, item_summary
            )

        if not use_cached_support:
            context_rows = torch.cat(
                (feature_summary, item_context[:, :support_size]), dim=1
            )

            # Item attention runs over rows separately for each feature group.
            context_sequence = context_rows.transpose(1, 2)
            retain_context = cache_support or item_context.shape[1] > support_size
            if output_support:
                updated_prefix = context_sequence + self._attention(
                    self.item_attention,
                    context_sequence,
                    cache_context=retain_context,
                )
                updated_prefix = updated_prefix.transpose(1, 2)
                summary_rows = updated_prefix[:, : self.feature_summary_count]
            else:
                self.item_attention.cache_context_projections(
                    context_sequence, first_head_only=True
                )
                updated_prefix = None
                summary_rows = feature_summary
            if retain_context and output_support:
                # Query cross-attention requests only the first context head.
                # Release the unused projected heads before advancing to the
                # next layer so a 50k-row cache remains practical.
                self.item_attention.retain_first_cached_context_head()
        else:
            assert self._cached_summary_rows is not None
            summary_rows = (
                self._cached_summary_rows[:, :, -1:, :]
                if label_group_only
                else self._cached_summary_rows
            )

        if use_cached_support:
            query_rows = (
                item_context
                if compact_cached_support
                else item_context[:, support_size:]
            )
            updated_support = None
        else:
            updated_support = (
                updated_prefix[:, self.feature_summary_count :]
                if output_support
                else item_context[:, :support_size]
            )
            query_rows = item_context[:, support_size:]

        if query_rows.shape[1]:
            # Cross-attention has no query rows in the context, so query rows
            # can share one batched context without interacting with each other.
            query_sequence = query_rows.transpose(1, 2)
            try:
                if self.item_attention.has_cached_context:
                    attended = self._attention(
                        self.item_attention,
                        query_sequence,
                        shared_context_head=True,
                        use_cached_context=True,
                    ).transpose(1, 2)
                else:
                    attended = self._attention(
                        self.item_attention,
                        query_sequence,
                        context_sequence,
                        shared_context_head=True,
                    ).transpose(1, 2)
            finally:
                if not cache_support and not use_cached_support:
                    self.item_attention.clear_cache()
            updated_query = query_rows + attended
        else:
            updated_query = query_rows

        if use_cached_support:
            if compact_cached_support:
                state = updated_query
            else:
                # Preserve the ordinary joined row count and query row positions.
                # Prefix values are padding from this point: every support-derived
                # cross-row value comes from the fitted layer cache.
                state = torch.cat((state[:, :support_size], updated_query), dim=1)
        else:
            assert updated_support is not None
            state = torch.cat((updated_support, updated_query), dim=1)
        feature_summary = summary_rows

        if (
            not output_support
            and (label_group_only or state.shape[2] >= 64)
        ):
            query_state = self._normalize(updated_query)
            query_state = self._feedforward(
                self.state_feedforward,
                query_state,
                feedforward_token_chunk,
            )
            query_state = self._normalize(query_state)
            if compact_cached_support:
                state = query_state
            elif use_cached_support:
                state = torch.cat((state[:, :support_size], query_state), dim=1)
            else:
                assert updated_support is not None
                state = torch.cat((updated_support, query_state), dim=1)
            if cache_support:
                self._cached_feature_input = feature_input.detach().clone()
                selected_summary = (
                    cache_summary_rows
                    if cache_summary_rows is not None
                    else summary_rows
                )
                self._cached_summary_rows = selected_summary.detach().clone()
                self._cached_support_rows = support_size
                if cache_support_to_cpu:
                    self.item_attention.offload_cached_context()
            return state, feature_summary, item_summary

        state = self._normalize(state)
        feature_summary = self._normalize(feature_summary)
        item_summary = self._normalize(item_summary)

        state = self._feedforward(
            self.state_feedforward,
            state,
            feedforward_token_chunk,
        )
        if not self.final_layer:
            assert self.feature_summary_feedforward is not None
            assert self.item_summary_feedforward is not None
            feature_summary = self._feedforward(
                self.feature_summary_feedforward,
                feature_summary,
                feedforward_token_chunk,
            )
            item_summary = self._feedforward(
                self.item_summary_feedforward,
                item_summary,
                feedforward_token_chunk,
            )
        state = self._normalize(state)
        feature_summary = self._normalize(feature_summary)
        item_summary = self._normalize(item_summary)

        if cache_support:
            self._cached_feature_input = feature_input.detach().clone()
            selected_summary = (
                cache_summary_rows if cache_summary_rows is not None else summary_rows
            )
            self._cached_summary_rows = selected_summary.detach().clone()
            self._cached_support_rows = support_size
            if cache_support_to_cpu:
                self.item_attention.offload_cached_context()

        return state, feature_summary, item_summary
