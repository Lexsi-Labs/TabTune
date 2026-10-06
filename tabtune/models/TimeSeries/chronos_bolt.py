"""Chronos-Bolt adapter: the only place TabTune talks to the Bolt API.

Chronos-Bolt patches the history, encodes it with a T5 encoder-decoder and reads
all nine trained quantiles off one forward pass. It is deterministic (no seed)
and quantile-native: the point forecast is the median, which upstream's
``predict_quantiles`` returns in the slot it calls ``mean``.

Levels inside the trained grid (0.1 ... 0.9) are read off exactly; others are
interpolated upstream, and levels outside it are clamped, which this adapter
warns about.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

from ._levels import warn_clamped_levels
from ._params import as_int, check_median_only, hub_kwargs, read_params
from .base import AdapterOutput, TrainingSpec, TSFMAdapter

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ...config.schemas import ForecastConfig
    from ...registry.TimeSeries import TimeSeriesModelSpec
    from ...TimeSeries.schema import TimeSeriesPanel

logger = logging.getLogger(__name__)

__all__ = ["ChronosBoltAdapter"]

_KNOWN_KEYS = ("batch_size", "dtype", "point_forecast", "revision", "local_files_only")
_DTYPES = ("float32", "bfloat16")


class ChronosBoltAdapter(TSFMAdapter):
    """Run Chronos-Bolt (``amazon/chronos-bolt-*``) behind the TSFM adapter interface.

    ``model_params`` accepted:
        batch_size: Series per forward pass (default 256). Bounds memory on
            large panels. Beyond the built-in 64-step horizon upstream expands
            the batch by the nine trained quantiles, so lower it for long
            horizons on big panels.
        dtype: ``"float32"`` or ``"bfloat16"``. Defaults to bfloat16 on CUDA,
            where upstream recommends it, and float32 elsewhere.
        point_forecast: Only ``"median"``. Bolt produces no sample paths, so
            there is no mean to take.

    Forecasts are deterministic; ``tuning_params={"seed": ...}`` has no effect
    here. Changing ``batch_size`` shifts results by about 1e-6 (torch reduction
    order).
    """

    point_forecast = "median"
    supported_dtypes = _DTYPES
    cuda_default_dtype = "bfloat16"
    # Attention only: the feed-forward is DenseReluDense.wi/wo or DenseGatedActDense.wi_0/wi_1/wo
    # depending on the checkpoint.
    lora_targets = (
        "SelfAttention.q",
        "SelfAttention.k",
        "SelfAttention.v",
        "SelfAttention.o",
        "EncDecAttention.q",
        "EncDecAttention.v",
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
        check_median_only(params, name=spec.name)

    def load(self) -> None:
        import torch

        from ..chronos import ChronosBoltPipeline

        logger.info(
            "[Chronos-Bolt] Loading %s (%s) on %s", self.checkpoint, self.dtype, self.device
        )
        # Load in float32: fine-tuned weights are saved as float32 and must not be rounded on restore.
        pipeline = ChronosBoltPipeline.from_pretrained(
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

        contexts = [torch.as_tensor(values, dtype=torch.float32) for values in panel.values]
        quantile_chunks: list[np.ndarray] = []
        median_chunks: list[np.ndarray] = []

        with torch.inference_mode():
            for start in range(0, len(contexts), self.batch_size):
                quantiles, median = self._model.predict_quantiles(
                    contexts[start : start + self.batch_size],
                    prediction_length=config.prediction_length,
                    quantile_levels=requested,
                    limit_prediction_length=False,
                )
                quantile_chunks.append(quantiles.float().numpy())
                median_chunks.append(median.float().numpy())

        point = np.concatenate(median_chunks, axis=0)
        quantiles_np = np.concatenate(quantile_chunks, axis=0) if levels else None
        return AdapterOutput(point=point, quantiles=quantiles_np)

    def embed(self, panel: TimeSeriesPanel) -> np.ndarray:
        """Mean of the encoder states over the patches that hold the series.

        ``encode`` returns the per-patch mask (1 if any step in the patch is
        observed), which excludes padding. The trailing ``[REG]`` state is
        dropped when the checkpoint uses one. Dimension is ``d_model``.
        """
        import torch

        net = self._model.model
        limit = int(net.chronos_config.context_length)
        vectors = []
        with torch.inference_mode():
            for start in range(0, len(panel), self.batch_size):
                chunk = [
                    np.asarray(v, dtype=np.float32)[-limit:]
                    for v in panel.values[start : start + self.batch_size]
                ]
                width = max(len(c) for c in chunk)
                padded = np.full((len(chunk), width), np.nan, dtype=np.float32)
                for i, series in enumerate(chunk):
                    padded[i, width - len(series) :] = series
                hidden, _, _, mask = net.encode(
                    context=torch.as_tensor(padded, device=self.device)
                )
                patches = hidden.shape[1] - 1 if net.chronos_config.use_reg_token else hidden.shape[1]
                hidden = hidden[:, :patches].float()
                weights = mask[:, :patches, None].to(hidden.dtype)
                pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
                vectors.append(pooled.cpu().numpy())
        return np.concatenate(vectors).astype(float)

    def finetune(self, panel: TimeSeriesPanel, spec: TrainingSpec) -> dict[str, Any]:
        import torch

        net = self._model.model
        horizon = int(net.chronos_config.prediction_length)
        spec = self.training_horizon(
            spec, horizon, reason=f"predicts {horizon} steps in one pass"
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
        return {"objective": "the checkpoint's own quantile loss over its nine quantiles", **report}

    def _network(self) -> Any:
        return self._model.model

    def _training_context(self, spec: TrainingSpec) -> int:
        limit = int(self._model.model.chronos_config.context_length)
        return min(spec.context_length or limit, limit)

    def _training_loss(self, torch: Any, context: np.ndarray, label: np.ndarray) -> Any:
        """The checkpoint's own training objective, computed by its own ``forward``.

        ``forward`` normalises the target with the context's ``loc_scale`` and
        right-pads targets shorter than ``prediction_length`` itself.
        """
        net = self._model.model
        ctx = torch.as_tensor(context, dtype=torch.float32, device=self.device)
        target = torch.as_tensor(label, dtype=torch.float32, device=self.device)
        return net(
            context=ctx,
            mask=~torch.isnan(ctx),
            target=target,
            target_mask=~torch.isnan(target),
        ).loss

    def quantile_range(self) -> tuple[float, float] | None:
        trained = list(getattr(self._model, "quantiles", []) or [])
        return (float(min(trained)), float(max(trained))) if trained else None

    def _warn_about_untrained_levels(self, levels: list[float]) -> None:
        """Warn that upstream clamps levels outside the trained grid."""
        warn_clamped_levels(
            list(getattr(self._model, "quantiles", []) or []), levels, model=self.spec.name
        )
