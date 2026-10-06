"""Chronos-2 adapter: the only place TabTune talks to the Chronos-2 API.

Chronos-2 is an encoder-only model whose group attention sees an item's target
variates, past-only covariates and known-future covariates in one pass. It is
deterministic (no seed) and quantile-native: the point forecast is the median,
which upstream's ``predict_quantiles`` returns in the slot it calls ``mean``.

The panel has one row per ``(item, target)``. An item's variates are forecast
jointly, so the adapter groups rows with
:meth:`~tabtune.TimeSeries.schema.TimeSeriesPanel.item_groups` and scatters the
per-item results back into row order.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

from ._levels import warn_clamped_levels
from ._params import as_bool, as_int, check_median_only, hub_kwargs, read_params
from .base import AdapterOutput, TrainingSpec, TSFMAdapter

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ...config.schemas import ForecastConfig
    from ...registry.TimeSeries import TimeSeriesModelSpec
    from ...TimeSeries.schema import TimeSeriesPanel

logger = logging.getLogger(__name__)

__all__ = ["Chronos2Adapter"]

_KNOWN_KEYS = ("batch_size", "dtype", "cross_learning", "point_forecast", "revision", "local_files_only")
_DTYPES = ("float32", "bfloat16")


class Chronos2Adapter(TSFMAdapter):
    """Run Chronos-2 (``amazon/chronos-2``) behind the TSFM adapter interface.

    ``model_params`` accepted:
        batch_size: Series per forward pass (default 256). Upstream counts
            *every* series fed to the model, targets and covariates alike, so
            an item with several variates and covariates uses more than one
            slot.
        dtype: ``"float32"`` or ``"bfloat16"``. Defaults to bfloat16 on CUDA,
            where upstream recommends it, and float32 elsewhere.
        cross_learning: When ``True`` the model attends across all items in a
            batch instead of forecasting each independently (default ``False``);
            results then depend on the batch size.
        point_forecast: Only ``"median"``. Chronos-2 produces no sample paths,
            so there is no mean to take.

    Forecasts are deterministic; ``tuning_params={"seed": ...}`` has no effect here.
    """

    point_forecast = "median"
    supported_dtypes = _DTYPES
    cuda_default_dtype = "bfloat16"
    default_train_context = 1024
    # Both attention sub-layers of a block (over time and across the group) are named self_attention.
    lora_targets = (
        "self_attention.q",
        "self_attention.k",
        "self_attention.v",
        "self_attention.o",
        "mlp.wi",
        "mlp.wo",
    )

    def __init__(
        self,
        spec: TimeSeriesModelSpec,
        *,
        checkpoint: str,
        device: str,
        model_params: Mapping[str, Any] | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            spec, checkpoint=checkpoint, device=device, model_params=model_params, seed=seed
        )
        params = self.model_params = read_params(
            self.model_params, name=spec.name, known=_KNOWN_KEYS
        )
        self.batch_size = as_int(params, "batch_size", 256, name=spec.name, minimum=1)
        self.dtype = self._resolve_dtype()
        self.cross_learning = as_bool(params, "cross_learning", False, name=spec.name)
        check_median_only(params, name=spec.name)

    def load(self) -> None:
        import torch

        from ..chronos import Chronos2Pipeline

        logger.info("[Chronos-2] Loading %s (%s) on %s", self.checkpoint, self.dtype, self.device)
        # Load in float32: fine-tuned weights are saved as float32 and must not be rounded on restore.
        pipeline = Chronos2Pipeline.from_pretrained(
            self.checkpoint, dtype=torch.float32, **hub_kwargs(self.model_params)
        )
        pipeline.model.to(self.device)
        self._restore_tuned(pipeline.model)
        pipeline.model.to(dtype=getattr(torch, self.dtype))
        pipeline.model.eval()
        self._model = pipeline

    def forecast(self, panel: TimeSeriesPanel, config: ForecastConfig) -> AdapterOutput:
        import torch

        levels = [float(q) for q in config.quantile_levels]
        # predict_quantiles needs at least one level; its "mean" return is the median.
        requested = levels or [0.5]
        self._warn_about_untrained_levels(levels)

        groups = panel.item_groups()
        inputs = [self._input_for(panel, rows) for rows in groups]

        with torch.inference_mode():
            quantiles, medians = self._model.predict_quantiles(
                inputs,
                prediction_length=config.prediction_length,
                quantile_levels=requested,
                batch_size=self.batch_size,
                cross_learning=self.cross_learning,
                limit_prediction_length=False,
            )

        # One entry per item: [n_variates, H, Q] quantiles and [n_variates, H] medians.
        point = np.empty((len(panel), config.prediction_length), dtype="float64")
        quantiles_np = np.empty(
            (len(panel), config.prediction_length, len(requested)), dtype="float64"
        )
        for rows, item_quantiles, item_median in zip(groups, quantiles, medians, strict=True):
            variates = len(rows)
            item_quantiles = item_quantiles.float().numpy().reshape(
                variates, config.prediction_length, len(requested)
            )
            item_median = item_median.float().numpy().reshape(variates, config.prediction_length)
            for variate, row in enumerate(rows):
                point[row] = item_median[variate]
                quantiles_np[row] = item_quantiles[variate]

        return AdapterOutput(point=point, quantiles=quantiles_np if levels else None)

    def embed(self, panel: TimeSeriesPanel) -> np.ndarray:
        """Mean of the encoder states over the context patches that hold the series.

        ``encode`` reports how many leading positions are context patches, which
        excludes the ``[REG]`` state and the future positions; the per-patch mask
        from ``_prepare_patched_context`` excludes padding. Rows are encoded
        independently (no ``group_ids``). Dimension is ``d_model``.
        """
        import torch

        net = self._model.model
        vectors = []
        with torch.inference_mode():
            for start in range(0, len(panel), self.batch_size):
                chunk = [
                    np.asarray(v, dtype=np.float32)
                    for v in panel.values[start : start + self.batch_size]
                ]
                width = max(len(c) for c in chunk)
                padded = np.full((len(chunk), width), np.nan, dtype=np.float32)
                for i, series in enumerate(chunk):
                    padded[i, width - len(series) :] = series
                context = torch.as_tensor(padded, device=self.device)
                _, mask, _ = net._prepare_patched_context(context=context)
                encoder_outputs, _, _, patches = net.encode(context=context, num_output_patches=1)
                hidden = encoder_outputs.last_hidden_state[:, :patches].float()
                weights = mask[:, :patches, None].to(hidden.dtype)
                pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
                vectors.append(pooled.cpu().numpy())
        return np.concatenate(vectors).astype(float)

    def finetune(self, panel: TimeSeriesPanel, spec: TrainingSpec) -> dict[str, Any]:
        import torch

        net = self._model.model
        limit = int(net.chronos_config.max_output_patches) * int(
            net.chronos_config.output_patch_size
        )
        spec = self.training_horizon(
            spec, limit, reason=f"predicts at most {limit} steps in one pass"
        )
        # AdamW on bfloat16 parameters is unsound, so train in float32 and cast back.
        net.float()
        try:
            report = super().finetune(panel, spec)
        finally:
            for p in net.parameters():
                p.requires_grad = False
            net.to(dtype=getattr(torch, self.dtype))
            net.eval()
        return {
            "objective": "the checkpoint's own quantile loss over its nine quantiles",
            "trained_on": "univariate windows without covariates",
            **report,
        }

    def _network(self) -> Any:
        return self._model.model

    def _training_context(self, spec: TrainingSpec) -> int:
        limit = int(self._model.model.chronos_config.context_length)
        return min(spec.context_length or self.default_train_context, limit)

    def _training_loss(self, torch: Any, context: np.ndarray, label: np.ndarray) -> Any:
        """The checkpoint's own training objective, computed by its own ``forward``.

        Masks for both the context and the future target are derived from NaN
        upstream, and ``group_ids`` is left unset so each window is independent.
        """
        net = self._model.model
        patch = int(net.chronos_config.output_patch_size)
        patches = min(
            math.ceil(label.shape[1] / patch), int(net.chronos_config.max_output_patches)
        )
        return net(
            context=torch.as_tensor(context, dtype=torch.float32, device=self.device),
            future_target=torch.as_tensor(label, dtype=torch.float32, device=self.device),
            num_output_patches=patches,
        ).loss

    def _input_for(self, panel: TimeSeriesPanel, rows: tuple[int, ...]) -> dict[str, Any]:
        """Build one upstream input dict for the panel rows of a single item.

        A single row stays a 1-D target; several rows stack into
        ``(n_variates, history)``. Covariates are read from the first row.
        """
        histories = [panel.values[row] for row in rows]
        target = histories[0] if len(rows) == 1 else np.stack(histories)

        prepared: dict[str, Any] = {"target": target}
        past = panel.past_covariates[rows[0]]
        if past:
            prepared["past_covariates"] = dict(past)
        future = panel.future_covariates[rows[0]]
        if future:
            prepared["future_covariates"] = dict(future)
        return prepared

    def quantile_range(self) -> tuple[float, float] | None:
        trained = list(getattr(self._model, "quantiles", []) or [])
        return (float(min(trained)), float(max(trained))) if trained else None

    def _warn_about_untrained_levels(self, levels: list[float]) -> None:
        """Warn that upstream clamps levels outside the trained grid."""
        warn_clamped_levels(
            list(getattr(self._model, "quantiles", []) or []), levels, model=self.spec.name
        )
