"""Toto 1.0 adapter: the only place TabTune talks to the Toto 1.0 API.

Toto 1.0 (``Datadog/Toto-Open-Base-1.0``) is sample-based: quantiles are order
statistics of the drawn sample paths, so any level is served exactly. The point
forecast is the median (default) or the mean of those paths. Pass
``tuning_params={"seed": ...}`` for reproducible output; a fixed seed reproduces
a forecast only for a fixed ``samples_per_batch``.

The model is time-aware: ``MaskedTimeseries`` carries POSIX timestamps (clamped
to int32, as upstream does) and the step length in seconds, both derived from the
panel. Known-future covariates go through upstream's exogenous path as the last
channels of the variate axis; on the released weights their effect is negligible.
Upstream rounds the context to the patch stride itself, so this adapter does not.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

from ...registry.errors import ConfigError
from ._params import as_bool, as_int, hub_kwargs, read_params
from .base import AdapterOutput, TrainingSpec, TSFMAdapter

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ...config.schemas import ForecastConfig
    from ...registry.TimeSeries import TimeSeriesModelSpec
    from ...TimeSeries.schema import TimeSeriesPanel

logger = logging.getLogger(__name__)

__all__ = ["Toto1Adapter"]

_KNOWN_KEYS = (
    "dtype",
    "revision",
    "local_files_only",
    "batch_size",
    "num_samples",
    "samples_per_batch",
    "use_kv_cache",
    "point_forecast",
)
_POINT_FORECASTS = ("median", "mean")

# Upstream's MaskedTimeseries declares timestamp_seconds as int32 and clamps to it.
_INT32_MIN = -(2**31)
_INT32_MAX = 2**31 - 1


class Toto1Adapter(TSFMAdapter):
    """Run Toto 1.0 (``Datadog/Toto-Open-Base-1.0``) behind the TSFM adapter interface.

    ``model_params`` accepted:
        batch_size: Items per forward pass (default 32). The effective tensor
            batch is this times ``samples_per_batch``, so the two multiply.
        num_samples: Sample paths drawn per series (default 256). Upstream
            recommends at least 128. ``None`` selects upstream's mean-only mode,
            which produces no sample paths and therefore no quantiles.
        samples_per_batch: Sample paths per forward pass (default 64). Must
            divide ``num_samples``. This is **not** only a memory knob: it
            changes which random numbers each path receives, so a fixed seed
            reproduces a forecast only for a fixed value here.
        use_kv_cache: Reuse transformer attention state across decoding steps
            (default ``True``, as upstream recommends). A pure speed setting.
        point_forecast: ``"median"`` (default) or ``"mean"``, both computed from
            the sample paths.

    No ``dtype``: the published checkpoint is float32 and upstream offers no
    precision setting.
    """

    respects_seed = True
    default_train_context = 1024
    # Attention only: the MLP is mlp.0/mlp.2 unfused but mlp.0.w12/mlp.0.w3 with fused SwiGLU.
    lora_targets = ("attention.wQKV", "attention.wO")

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
        self.batch_size = as_int(params, "batch_size", 32, name=spec.name, minimum=1)
        self._resolve_dtype()

        # None is upstream's mean-only mode, not "unset".
        self.num_samples = (
            None
            if "num_samples" in params and params["num_samples"] is None
            else as_int(params, "num_samples", 256, name=spec.name, minimum=1)
        )
        self.samples_per_batch = as_int(
            params, "samples_per_batch", 64, name=spec.name, minimum=1
        )
        if self.num_samples is not None and self.num_samples % self.samples_per_batch:
            raise ConfigError(
                f"Toto1 num_samples ({self.num_samples}) must be a multiple of "
                f"samples_per_batch ({self.samples_per_batch}): upstream draws the "
                f"paths in equal batches and asserts this. Pick a samples_per_batch "
                f"that divides num_samples, such as "
                f"{self._largest_divisor(self.num_samples)}."
            )

        self.use_kv_cache = as_bool(params, "use_kv_cache", True, name=spec.name)

        requested_point = params.get("point_forecast")
        if self.num_samples is None:
            if requested_point == "median":
                raise ConfigError(
                    "Toto1 cannot produce a median with num_samples=None: that is "
                    "upstream's mean-only mode, which draws no sample paths. Set a "
                    "num_samples, or use point_forecast='mean'."
                )
            self.point_forecast = requested_point or "mean"
        else:
            self.point_forecast = requested_point or "median"
        if self.point_forecast not in _POINT_FORECASTS:
            raise ConfigError(
                f"Toto1 point_forecast must be one of {_POINT_FORECASTS}, "
                f"got {self.point_forecast!r}"
            )

    @staticmethod
    def _largest_divisor(num_samples: int) -> int:
        """The largest divisor of ``num_samples`` at most 64, for the error hint."""
        return next(n for n in range(min(64, num_samples), 0, -1) if num_samples % n == 0)

    def load(self) -> None:
        try:
            with warnings.catch_warnings():
                # Upstream warns once per module that xformers' fused kernels are absent.
                warnings.simplefilter("ignore", ImportWarning)
                from ..toto.toto1 import MaskedTimeseries, Toto, TotoForecaster
        except ImportError as exc:
            raise self.missing_extra_error(exc, "rotary_embedding_torch") from exc

        logger.info("[Toto1] Loading %s on %s", self.checkpoint, self.device)
        model = Toto.from_pretrained(self.checkpoint, **hub_kwargs(self.model_params))
        model.to(self.device)
        self._restore_tuned(model.model)
        model.eval()
        self._model = TotoForecaster(model.model)
        self._masked_timeseries = MaskedTimeseries
        self._patch_stride = int(model.model.patch_embed.stride)

    def forecast(self, panel: TimeSeriesPanel, config: ForecastConfig) -> AdapterOutput:
        import torch

        horizon = config.prediction_length
        levels = [float(q) for q in config.quantile_levels]
        if levels and self.num_samples is None:
            raise ConfigError(
                f"Toto1 cannot serve quantile levels {levels} with num_samples=None: "
                f"that is upstream's mean-only mode, which draws no sample paths to "
                f"take quantiles from. Set a num_samples, or request no quantile levels."
            )
        groups = panel.item_groups()
        point = np.empty((len(panel), horizon), dtype="float64")
        quantiles_np = (
            np.empty((len(panel), horizon, len(levels)), dtype="float64") if levels else None
        )

        known = self._known_names(panel)
        past_only = [name for name in panel.covariate_names if name not in known]

        with self._seeded_rng(torch), torch.inference_mode():
            for start in range(0, len(groups), self.batch_size):
                chunk = groups[start : start + self.batch_size]
                inputs, future_ev = self._batch_for(panel, chunk, past_only, known, horizon)
                output = self._model.forecast(
                    inputs,
                    prediction_length=horizon,
                    num_samples=self.num_samples,
                    samples_per_batch=self.samples_per_batch,
                    use_kv_cache=self.use_kv_cache,
                    future_exogenous_variables=future_ev,
                )

                if self.point_forecast == "median":
                    batch_point = output.median
                else:
                    batch_point = output.mean
                batch_point = batch_point.float().cpu().numpy().astype("float64")
                batch_quantiles = (
                    np.stack(
                        [output.quantile(level).float().cpu().numpy() for level in levels],
                        axis=-1,
                    ).astype("float64")
                    if levels
                    else None
                )

                for entry, item_rows in enumerate(chunk):
                    for variate, row in enumerate(item_rows):
                        point[row] = batch_point[entry, variate]
                        if quantiles_np is not None:
                            quantiles_np[row] = batch_quantiles[entry, variate]

        return AdapterOutput(point=point, quantiles=quantiles_np)

    def embed(self, panel: TimeSeriesPanel) -> np.ndarray:
        """Mean of the backbone states over the observed steps of the series.

        ``backbone`` stops short of ``output_distribution`` and unembeds to one
        state per time step. Targets only. Dimension is ``embed_dim``.
        """
        import torch

        net = self._model.model
        stride = self._patch_stride
        # Rows are bucketed by their own patch-rounded width: extra leading padding would
        # shift the rotary positions and make a vector depend on its batch mates.
        buckets: dict[int, list[int]] = {}
        for row, values in enumerate(panel.values):
            width = len(values) + (-len(values) % stride)
            buckets.setdefault(max(width, stride), []).append(row)

        vectors: dict[int, np.ndarray] = {}
        with torch.inference_mode():
            for width, rows in buckets.items():
                for start in range(0, len(rows), self.batch_size):
                    chunk = rows[start : start + self.batch_size]
                    padded = np.full((len(chunk), width), np.nan, dtype=np.float32)
                    for i, row in enumerate(chunk):
                        series = np.asarray(panel.values[row], dtype=np.float32)
                        padded[i, width - len(series) :] = series
                    observed = np.isfinite(padded)
                    values = torch.as_tensor(
                        np.nan_to_num(padded), dtype=torch.float32, device=self.device
                    )[:, None, :]
                    mask = torch.as_tensor(observed, device=self.device)[:, None, :]
                    hidden, _, _ = net.backbone(
                        inputs=values,
                        input_padding_mask=mask,
                        id_mask=torch.zeros_like(values),
                    )
                    hidden = hidden[:, 0].float()
                    weights = mask[:, 0, :, None].to(hidden.dtype)
                    pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
                    pooled = pooled.cpu().numpy()
                    for i, row in enumerate(chunk):
                        vectors[row] = pooled[i]
        return np.stack([vectors[row] for row in range(len(panel))]).astype(float)

    def finetune(self, panel: TimeSeriesPanel, spec: TrainingSpec) -> dict[str, Any]:
        net = self._model.model
        try:
            report = super().finetune(panel, spec)
        finally:
            for p in net.parameters():
                p.requires_grad = False
            net.eval()
        return {
            "objective": "next-patch negative log-likelihood under the Student-T mixture",
            "trained_on": "univariate windows without covariates",
            **report,
        }

    def _network(self) -> Any:
        return self._model.model

    def _training_context(self, spec: TrainingSpec) -> int:
        stride = self._patch_stride
        limit = self.spec.max_context or self.default_train_context
        context = min(spec.context_length or self.default_train_context, limit)
        return max(stride, context - context % stride)

    def _training_loss(self, torch: Any, context: np.ndarray, label: np.ndarray) -> Any:
        """Toto's pretraining objective: the likelihood of the patch after each position.

        The state at step ``i`` parameterises step ``i + stride``, so targets are
        shifted by one patch and scaled with the ``loc``/``scale`` at ``i``; no
        ``context_scale`` division applies on top. ``MixtureSameFamily`` cannot be
        indexed, so the target is built at full width.
        """
        net = self._model.model
        stride = self._patch_stride
        window = np.concatenate([context, label], axis=1)
        pad = -window.shape[1] % stride
        if pad:
            window = np.concatenate(
                [np.full((len(window), pad), np.nan, dtype=np.float32), window], axis=1
            )
        observed = np.isfinite(window)

        values = torch.as_tensor(
            np.nan_to_num(window), dtype=torch.float32, device=self.device
        )[:, None, :]
        mask = torch.as_tensor(observed, device=self.device)[:, None, :]
        output = net(
            inputs=values,
            input_padding_mask=mask,
            id_mask=torch.zeros_like(values),
        )

        steps = values.shape[-1]
        future = torch.zeros_like(values)
        future[..., : steps - stride] = values[..., stride:]
        scored = torch.zeros_like(mask)
        scored[..., : steps - stride] = mask[..., stride:]
        # loc and scale come from the no_grad scaler, so the target carries no gradient.
        target = (future - output.loc) / output.scale
        log_prob = output.distribution.log_prob(target) * scored
        return -log_prob.sum() / scored.sum().clamp_min(1.0)

    @staticmethod
    def _known_names(panel: TimeSeriesPanel) -> list[str]:
        """Covariate names that carry values over the horizon, in panel order."""
        if not panel.future_covariates:
            return []
        future = panel.future_covariates[0]
        return [name for name in panel.covariate_names if name in future]

    def _batch_for(
        self,
        panel: TimeSeriesPanel,
        groups: tuple[tuple[int, ...], ...],
        past_only: list[str],
        known: list[str],
        horizon: int,
    ) -> tuple[Any, Any]:
        """Build one chunk's ``MaskedTimeseries`` and its known-future values.

        Channels are ``[targets..., past-only covariates..., known-future
        covariates...]``: upstream requires exogenous variables in the last
        ``num_exogenous_variables`` channels. Histories are left-aligned to the
        chunk's longest with the filler marked unobserved, and not rounded to a
        patch boundary; upstream's ``forecast`` pads itself.
        """
        import torch

        targets = len(groups[0])
        variates = targets + len(past_only) + len(known)
        context = max(len(panel.values[rows[0]]) for rows in groups)

        values = np.zeros((len(groups), variates, context), dtype="float64")
        observed = np.zeros((len(groups), variates, context), dtype=bool)
        stamps = np.zeros((len(groups), variates, context), dtype="int64")
        intervals = np.zeros((len(groups), variates), dtype="int64")
        future_ev = (
            np.zeros((len(groups), len(known), horizon), dtype="float64") if known else None
        )

        step = _freq_to_seconds(panel.freq)
        for entry, item_rows in enumerate(groups):
            history = panel.values[item_rows[0]]
            offset = context - len(history)
            channels = [panel.values[row] for row in item_rows]
            past = panel.past_covariates[item_rows[0]] if panel.past_covariates else {}
            channels += [np.asarray(past[name], dtype="float64") for name in past_only]
            channels += [np.asarray(past[name], dtype="float64") for name in known]

            for channel, series in enumerate(channels):
                values[entry, channel, offset:] = np.nan_to_num(series, copy=True)
                observed[entry, channel, offset:] = ~np.isnan(series)

            # Upstream extends the stamps through the horizon itself as last stamp + interval.
            origin = panel.last_timestamps[item_rows[0]].value // 10**9
            row_stamps = origin - step * np.arange(len(history) - 1, -1, -1, dtype="int64")
            stamps[entry, :, offset:] = np.clip(row_stamps, _INT32_MIN, _INT32_MAX)
            intervals[entry, :] = step

            if future_ev is not None:
                future = panel.future_covariates[item_rows[0]]
                for channel, name in enumerate(known):
                    future_ev[entry, channel] = np.asarray(future[name], dtype="float64")

        inputs = self._masked_timeseries(
            series=torch.as_tensor(values, dtype=torch.float32, device=self.device),
            padding_mask=torch.as_tensor(observed, dtype=torch.bool, device=self.device),
            id_mask=torch.zeros(values.shape, dtype=torch.int, device=self.device),
            timestamp_seconds=torch.as_tensor(stamps, dtype=torch.int, device=self.device),
            time_interval_seconds=torch.as_tensor(
                intervals, dtype=torch.int, device=self.device
            ),
            num_exogenous_variables=len(known),
        )
        future_tensor = (
            torch.as_tensor(future_ev, dtype=torch.float32, device=self.device)
            if future_ev is not None
            else None
        )
        return inputs, future_tensor



def _freq_to_seconds(freq: str) -> int:
    """Seconds per step of a pandas frequency, mirroring ``TotoPredictor.freq_to_seconds``
    in upstream's (not vendored) ``inference/gluonts_predictor.py``.
    """
    from pandas.tseries.frequencies import to_offset

    offset = to_offset(freq)
    try:
        return int(offset.nanos // 10**9)
    except ValueError:
        pass

    import pandas as pd

    # Upstream's approximations, reproduced exactly: `n` applies only to Week, "2MS" is one month.
    day = 24 * 60 * 60
    if isinstance(offset, pd.offsets.Week):
        return int(offset.n * 7 * day)
    if isinstance(offset, (pd.offsets.MonthBegin, pd.offsets.MonthEnd)):
        return 30 * day
    if isinstance(offset, (pd.offsets.QuarterBegin, pd.offsets.QuarterEnd)):
        return 90 * day
    if isinstance(offset, (pd.offsets.YearBegin, pd.offsets.YearEnd)):
        return int(365.25 * day)
    raise ConfigError(
        f"Toto1 needs the step length of the series frequency in seconds, and "
        f"cannot derive one for {freq!r} ({type(offset).__name__}). Upstream handles "
        f"fixed frequencies plus weeks, months, quarters and years."
    )
