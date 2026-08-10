"""Task adapters joining the tabular model's encoder boundaries."""

from __future__ import annotations

import torch
from torch import nn

from ..config import ModelConfig, RegressionConfig
from ..runtime import FULL_FEATURE_ATTENTION_ROWS
from .encoders import FeatureEncoder, LabelEncoder, RegressionLabelEncoder
from .memory import PeakMemoryEstimate, PeakMemoryMode, estimated_peak
from .transformer import CrossAxisSummaryTransformer


class ClassificationModel(nn.Module):
    """Assemble feature, label, and cross-axis summary modules for classification."""

    supports_query_chunk_cache = True

    def __init__(
        self,
        config: ModelConfig,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self._initialize(config, device=device, dtype=dtype)

    def _initialize(
        self,
        config: ModelConfig,
        *,
        device: torch.device | str | None,
        dtype: torch.dtype | None,
        learned_head_scale: bool = True,
        initial_head_scale: float = 0.43,
        output_width: int | None = None,
        decoder_hidden_width: int | None = None,
    ) -> None:
        if not isinstance(config, ModelConfig):
            raise TypeError("config must be a ModelConfig")
        if dtype is not None and (
            not isinstance(dtype, torch.dtype) or not dtype.is_floating_point
        ):
            raise TypeError("dtype must be a floating torch dtype or None")

        # Validate the device before allocating any children.
        resolved_device = None if device is None else torch.device(device)
        self.feature_encoder = FeatureEncoder(
            config.columns_per_group, config.width, bias=False
        )
        self.label_encoder = LabelEncoder(config.width, bias=True)
        self.transformer = CrossAxisSummaryTransformer(
            config,
            learned_head_scale=learned_head_scale,
            initial_head_scale=initial_head_scale,
            output_width=output_width,
            decoder_hidden_width=decoder_hidden_width,
            device=resolved_device,
            dtype=dtype,
        )
        self.class_capacity = (
            config.class_capacity if output_width is None else output_width
        )
        self.columns_per_group = config.columns_per_group
        self.config = config
        self.to(device=resolved_device, dtype=dtype)

    def forward(
        self,
        x_support: torch.Tensor,
        y_support: torch.Tensor,
        x_query: torch.Tensor,
        *,
        feedforward_token_chunk: int,
        feature_attention_row_chunk: int = FULL_FEATURE_ATTENTION_ROWS,
        query_chunk_size: int | None = None,
        cache_support: bool = False,
        cache_support_to_cpu: bool = False,
        use_cached_support: bool = False,
        compact_cached_support: bool = False,
        trusted_internal_inputs: bool = False,
    ) -> torch.Tensor:
        if not isinstance(trusted_internal_inputs, bool):
            raise TypeError("trusted_internal_inputs must be bool")
        if not trusted_internal_inputs:
            self._validate_inputs(
                x_support,
                y_support,
                x_query,
                feedforward_token_chunk=feedforward_token_chunk,
                feature_attention_row_chunk=feature_attention_row_chunk,
                query_chunk_size=query_chunk_size,
                cache_support=cache_support,
                cache_support_to_cpu=cache_support_to_cpu,
                use_cached_support=use_cached_support,
                compact_cached_support=compact_cached_support,
            )
        parameter = next(self.parameters())
        x_support = x_support.to(device=parameter.device)
        y_support = y_support.to(device=parameter.device)
        x_query = x_query.to(device=parameter.device)
        ensemble_size, support_size, feature_count = x_support.shape
        query_size = x_query.shape[1]
        row_count = support_size + query_size

        joined = torch.cat((x_support, x_query), dim=1)
        remainder = feature_count % self.columns_per_group
        if remainder:
            padding = joined.new_zeros(
                ensemble_size, row_count, self.columns_per_group - remainder
            )
            joined = torch.cat((joined, padding), dim=2)
        group_count = joined.shape[2] // self.columns_per_group
        grouped = joined.reshape(
            ensemble_size, row_count, group_count, self.columns_per_group
        )

        # FeatureEncoder's public boundary is row-major, so members are kept
        # separate and then restored to batch-first transformer form.
        feature_states = [
            self.feature_encoder(grouped[index], support_size)
            for index in range(ensemble_size)
        ]
        feature_state = torch.stack(feature_states, dim=0)

        labels = y_support.transpose(0, 1).unsqueeze(-1)
        placeholders = y_support.new_full((query_size, ensemble_size, 1), float("nan"))
        label_state = self.label_encoder(
            torch.cat((labels, placeholders), dim=0), support_size
        ).transpose(0, 1)
        state = torch.cat((feature_state, label_state.unsqueeze(2)), dim=2)
        if cache_support:
            # Produce the first query chunk through the ordinary joined shape
            # while retaining its support context for equal-sized chunks.
            return self.transformer(
                state,
                support_size,
                feedforward_token_chunk=feedforward_token_chunk,
                feature_attention_row_chunk=feature_attention_row_chunk,
                output="standard",
                cache_support=True,
                cache_support_to_cpu=cache_support_to_cpu,
                query_chunk_size=query_chunk_size,
            )
        if use_cached_support:
            return self.transformer(
                state,
                support_size,
                feedforward_token_chunk=feedforward_token_chunk,
                feature_attention_row_chunk=feature_attention_row_chunk,
                output="standard",
                use_cached_support=True,
                compact_cached_support=compact_cached_support,
                query_chunk_size=query_chunk_size,
            )
        return self.transformer(
            state,
            support_size,
            feedforward_token_chunk=feedforward_token_chunk,
            feature_attention_row_chunk=feature_attention_row_chunk,
            output="standard",
            query_chunk_size=query_chunk_size,
        )

    def clear_support_cache(self) -> None:
        self.transformer.clear_cache()

    def load_support_cache(self, device: torch.device | str) -> None:
        """Reload an offloaded support cache onto ``device`` before the query phase."""
        self.transformer.load_cached_context(device)

    def estimated_peak(
        self,
        *,
        support_rows: int,
        query_rows: int,
        feature_count: int,
        members: int,
        feedforward_token_chunk: int,
        mode: PeakMemoryMode | str,
        max_vram_bytes: int,
        feature_attention_row_chunk: int = FULL_FEATURE_ATTENTION_ROWS,
        baseline_allocated_bytes: int = 0,
        baseline_reserved_bytes: int = 0,
        free_bytes: int | None = None,
    ) -> PeakMemoryEstimate:
        """Return the shape-derived peak for one inference phase."""
        parameter = next(self.parameters())
        return estimated_peak(
            self.config,
            support_rows=support_rows,
            query_rows=query_rows,
            feature_count=feature_count,
            members=members,
            feedforward_token_chunk=feedforward_token_chunk,
            feature_attention_row_chunk=feature_attention_row_chunk,
            dtype=parameter.dtype,
            mode=mode,
            max_vram_bytes=max_vram_bytes,
            baseline_allocated_bytes=baseline_allocated_bytes,
            baseline_reserved_bytes=baseline_reserved_bytes,
            free_bytes=free_bytes,
            output_width=self.class_capacity,
            decoder_hidden_width=(
                self.transformer.classification_heads["standard"][0].out_features
            ),
        )

    @staticmethod
    def can_cache_support(x_support: torch.Tensor) -> bool:
        """Whether query rows cannot alter support feature encoding.

        FeatureEncoder packs varying coordinates and rescales them by the
        active coordinate count.  Reuse is safe when every coordinate is
        already both retained and active from support rows alone; query rows
        can then change neither decision.
        """
        if not isinstance(x_support, torch.Tensor) or x_support.ndim != 3:
            return False
        if x_support.shape[1] <= 1:
            return False
        with torch.no_grad():
            finite = torch.isfinite(x_support)
            counts = finite.sum(dim=1).clamp_min(1)
            means = torch.where(finite, x_support, 0).sum(dim=1) / counts
            replaced = torch.where(finite, x_support, means[:, None, :])
            centered = replaced - replaced.mean(dim=1, keepdim=True)
            deviations = replaced.std(dim=1, correction=1, keepdim=True)
            deviations = torch.where(
                deviations == 0, torch.ones_like(deviations), deviations
            )
            normalized = (centered / (deviations + 1e-16)).clamp(-100, 100)
            normalized = torch.nan_to_num(
                normalized, nan=0.0, posinf=100.0, neginf=-100.0
            )
            retained = (x_support[:, 1:] != x_support[:, :1]).any(dim=1)
            active = (normalized[:, 1:] != normalized[:, :1]).any(dim=1)
            return bool((retained & active).all())

    def can_cache_query_chunks(
        self, x_support: torch.Tensor, x_query: torch.Tensor, query_rows: int
    ) -> bool:
        """Whether every query chunk selects the same feature groups.

        For one-coordinate groups, support encoding can depend on query values
        only through the retained-group mask. Comparing that mask across chunks
        safely admits common support-constant columns without forcing all such
        datasets down the uncached path.
        """
        if (
            not isinstance(x_support, torch.Tensor)
            or not isinstance(x_query, torch.Tensor)
            or x_support.ndim != 3
            or x_query.ndim != 3
            or x_support.shape[0] != x_query.shape[0]
            or x_support.shape[2] != x_query.shape[2]
            or x_support.shape[1] <= 1
            or self.columns_per_group != 1
            or isinstance(query_rows, bool)
            or not isinstance(query_rows, int)
            or query_rows <= 0
        ):
            return False
        if x_query.shape[1] <= query_rows:
            return False
        with torch.no_grad():
            first = x_support[:, :1]
            support_retained = (x_support[:, 1:] != first).any(dim=1)
            expected: torch.Tensor | None = None
            # Include a short remainder: its retained-feature mask must agree
            # with the cache-building chunk just like every full chunk.
            for start in range(0, x_query.shape[1], query_rows):
                query_retained = (
                    x_query[:, start : start + query_rows] != first
                ).any(dim=1)
                signature = support_retained | query_retained
                if expected is None:
                    expected = signature
                elif not torch.equal(signature, expected):
                    return False
            return True

    @staticmethod
    def _validate_inputs(
        x_support: object,
        y_support: object,
        x_query: object,
        *,
        feedforward_token_chunk: object,
        query_chunk_size: object,
        feature_attention_row_chunk: object = FULL_FEATURE_ATTENTION_ROWS,
        cache_support: object = False,
        cache_support_to_cpu: object = False,
        use_cached_support: object = False,
        compact_cached_support: object = False,
    ) -> None:
        if not all(isinstance(value, torch.Tensor) for value in (x_support, y_support, x_query)):
            raise TypeError("model inputs must be torch tensors")
        assert isinstance(x_support, torch.Tensor)
        assert isinstance(y_support, torch.Tensor)
        assert isinstance(x_query, torch.Tensor)
        if x_support.dtype not in (torch.float32, torch.float64):
            raise TypeError("feature dtype must be float32 or float64")
        if x_query.dtype not in (torch.float32, torch.float64):
            raise TypeError("feature dtype must be float32 or float64")
        if y_support.dtype not in (torch.float32, torch.float64):
            raise TypeError("label dtype must be float32 or float64")
        if x_support.ndim != 3 or x_query.ndim != 3 or y_support.ndim != 2:
            raise ValueError("model inputs have invalid ranks")
        e, s, f = x_support.shape
        if e <= 0 or s <= 0 or f <= 0 or x_query.shape[1] <= 0:
            raise ValueError("model input axes must be nonempty")
        if x_query.shape[0] != e or x_query.shape[2] != f or y_support.shape != (e, s):
            raise ValueError("model input shapes do not agree")
        if x_support.device != x_query.device or x_support.device != y_support.device:
            raise ValueError("model inputs must share a device")
        if x_support.dtype != x_query.dtype:
            raise ValueError("feature tensors must share a dtype")
        if not bool(torch.isfinite(y_support).all()):
            raise ValueError("support labels must be finite")
        if not bool((y_support == torch.trunc(y_support)).all()):
            raise ValueError("support labels must be integral")
        if not bool((y_support >= 0).all()):
            raise ValueError("support labels must be non-negative")
        if query_chunk_size is not None and (
            isinstance(query_chunk_size, bool)
            or not isinstance(query_chunk_size, int)
            or query_chunk_size <= 0
        ):
            raise ValueError("query_chunk_size must be a positive integer or None")
        if (
            isinstance(feedforward_token_chunk, bool)
            or not isinstance(feedforward_token_chunk, int)
            or feedforward_token_chunk <= 0
        ):
            raise ValueError("feedforward_token_chunk must be a positive integer")
        if (
            isinstance(feature_attention_row_chunk, bool)
            or not isinstance(feature_attention_row_chunk, int)
            or feature_attention_row_chunk <= 0
        ):
            raise ValueError("feature_attention_row_chunk must be a positive integer")
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


class RegressionModel(ClassificationModel):
    """Classification-core model with a direct quantile regression readout."""

    def __init__(
        self,
        config: ModelConfig,
        regression: RegressionConfig,
        *,
        device=None,
        dtype=None,
        fs_cfg=None,
    ) -> None:
        if fs_cfg is not None:
            raise ValueError("fs_cfg is unavailable for regression")
        if not isinstance(regression, RegressionConfig):
            raise TypeError("regression must be a RegressionConfig")
        nn.Module.__init__(self)
        self._initialize(
            config,
            learned_head_scale=False,
            initial_head_scale=regression.fixed_head_scale,
            output_width=regression.quantile_count,
            decoder_hidden_width=regression.decoder_hidden_width,
            device=device,
            dtype=dtype,
        )
        parameter = next(self.parameters())
        self.label_encoder = RegressionLabelEncoder(config.width, bias=True).to(
            device=parameter.device,
            dtype=parameter.dtype,
        )
        levels = torch.linspace(
            1 / (regression.quantile_count + 1),
            regression.quantile_count / (regression.quantile_count + 1),
            regression.quantile_count,
            device=parameter.device,
            dtype=torch.float32,
        )
        self.register_buffer("quantile_levels", levels)
        self.regression_config = regression
        self.output_width = regression.quantile_count

    @property
    def median_index(self) -> int:
        return self.regression_config.quantile_count // 2

    @staticmethod
    def _validate_inputs(x_support, y_support, x_query, **options) -> None:
        if not all(isinstance(value, torch.Tensor) for value in (x_support, y_support, x_query)):
            raise TypeError("model inputs must be torch tensors")
        if x_support.ndim != 3 or x_query.ndim != 3 or y_support.ndim != 2:
            raise ValueError("model inputs have invalid ranks")
        members, rows, width = x_support.shape
        if members < 1 or rows < 1 or width < 1 or x_query.shape[1] < 1:
            raise ValueError("model input axes must be nonempty")
        if x_query.shape[0] != members or x_query.shape[2] != width or y_support.shape != (members, rows):
            raise ValueError("model input shapes do not agree")
        if not all(value.is_floating_point() for value in (x_support, y_support, x_query)):
            raise TypeError("model inputs must have floating dtypes")
        if not bool(torch.isfinite(y_support).all()):
            raise ValueError("regression targets must be finite")
        chunk = options.get("feedforward_token_chunk")
        if isinstance(chunk, bool) or not isinstance(chunk, int) or chunk <= 0:
            raise ValueError("feedforward_token_chunk must be positive")
        row_chunk = options.get(
            "feature_attention_row_chunk", FULL_FEATURE_ATTENTION_ROWS
        )
        if (
            isinstance(row_chunk, bool)
            or not isinstance(row_chunk, int)
            or row_chunk <= 0
        ):
            raise ValueError("feature_attention_row_chunk must be positive")


__all__ = ["ClassificationModel", "RegressionModel"]


def build_model(manifest, *, device=None, dtype=None, fs_cfg=None):
    """Construct the model selected explicitly by an inference manifest."""
    from ..config import InferenceManifest

    if not isinstance(manifest, InferenceManifest):
        raise TypeError("manifest must be an InferenceManifest")
    if manifest.task == "regression":
        if fs_cfg is not None:
            raise ValueError("fs_cfg is unavailable for regression")
        return RegressionModel(
            manifest.model,
            manifest.regression,
            device=device,
            dtype=dtype,
        )
    return ClassificationModel(manifest.model, device=device, dtype=dtype)


create_model = build_model
build_model_from_manifest = build_model
create_model_from_manifest = build_model
model_from_manifest = build_model
__all__ = [
    "ClassificationModel",
    "RegressionModel",
    "build_model",
    "create_model",
    "build_model_from_manifest",
    "create_model_from_manifest",
    "model_from_manifest",
]
