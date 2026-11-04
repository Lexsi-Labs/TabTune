"""T0 adapter (model code vendored in :mod:`tabtune.models.t0`).

T0 (The Forecasting Company, 2026) is a patch transformer with factorised time
and group attention, in the line of Toto and Chronos-2. It forecasts an item's
targets jointly and conditions on known-future covariates. Past-only
covariates go in the context as extra variates, as upstream documents: they
are attended to and forecast with the targets, and their forecasts are
dropped. Quantile levels between the trained ones are interpolated and levels
beyond them extrapolated on exponential tails, so any level in (0, 1) is
served.

Items of different lengths are batched with ``TimeSeries.batch``, which pads on
the left with ``PAD`` cells the model does not attend to. The four inference
adjustments upstream recommends for the published scores (leading-missing
trimming, symmetric averaging, non-negative clipping, integer snapping) are on
by default.

Fine-tuning trains univariate windows with one differentiable pass over the
rollout buffer, the window upstream's ``_predict_step`` (and its LoRA
notebook) predicts from, with the pinball loss over the native quantiles
taken in the scaler's normalised space.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

from ...registry.errors import ConfigError
from ._params import as_bool, as_int, hub_kwargs, read_params
from . import _training
from .base import AdapterOutput, TrainingSpec, TSFMAdapter

if TYPE_CHECKING:
    from ...config.schemas import ForecastConfig
    from ...registry.TimeSeries import TimeSeriesModelSpec
    from ...TimeSeries.schema import TimeSeriesPanel

logger = logging.getLogger(__name__)

__all__ = ["T0Adapter"]

_ADJUSTMENTS = ("trim_leading_missing", "symmetric_averaging", "clip_nonnegative", "snap_integers")
_KNOWN_KEYS = ("batch_size", *_ADJUSTMENTS, "revision", "local_files_only", "dtype")


class T0Adapter(TSFMAdapter):
    """Run T0 (``theforecastingcompany/t0-alpha``, ``-beta``, ``-gamma``) behind the adapter interface.

    ``model_params`` accepted:
        batch_size: Items per forward pass (default 64).
        trim_leading_missing, symmetric_averaging, clip_nonnegative,
            snap_integers: upstream's inference adjustments, each using only
            the series' own history. All default to ``True``, the setting
            behind the published scores.
        revision, local_files_only: Hugging Face download options.

    Forecasts are deterministic; ``tuning_params={"seed": ...}`` has no effect here.
    """

    point_forecast = "median"
    refuses_other_dtypes = True
    default_train_context = 512
    lora_targets = ("wQKV", "wO", "mlp.0", "mlp.2")

    def __init__(
        self,
        spec: TimeSeriesModelSpec,
        *,
        checkpoint: str,
        device: str,
        model_params: Mapping[str, Any] | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__(spec, checkpoint=checkpoint, device=device, model_params=model_params, seed=seed)
        params = self.model_params = read_params(self.model_params, name=spec.name, known=_KNOWN_KEYS)
        self._resolve_dtype()
        self.batch_size = as_int(params, "batch_size", 64, name=spec.name, minimum=1)
        self.adjustments = {key: as_bool(params, key, True, name=spec.name) for key in _ADJUSTMENTS}

    def load(self) -> None:
        from ..t0 import T0Forecaster

        logger.info("[T0] Loading %s on %s", self.checkpoint, self.device)
        model = T0Forecaster.from_pretrained(self.checkpoint, **hub_kwargs(self.model_params))
        self._restore_tuned(model)
        model.to(self.device)
        model.eval()
        self._model = model

    @property
    def native_levels(self) -> tuple[float, ...]:
        return tuple(round(float(q), 6) for q in self._model.config.quantile_levels)

    def forecast(self, panel: TimeSeriesPanel, config: ForecastConfig) -> AdapterOutput:
        import torch

        from ..t0 import TimeSeries

        horizon = int(config.prediction_length)
        levels = [float(q) for q in config.quantile_levels]
        requested = sorted(set(levels) | {0.5})
        median = requested.index(0.5)
        groups = panel.item_groups()
        point = np.empty((len(panel), horizon))
        quantiles = np.empty((len(panel), horizon, len(levels))) if levels else None
        for start in range(0, len(groups), self.batch_size):
            chunk = groups[start : start + self.batch_size]
            series, widths, target_rows = [], [], []
            offset = 0
            for rows in chunk:
                item, n_rows = self._series(panel, rows, horizon)
                series.append(item)
                widths.append(item.seq_len - horizon)
                target_rows.append(list(range(offset, offset + len(rows))))
                offset += n_rows
            batch = TimeSeries.batch(series)
            with torch.inference_mode():
                out = self._model.predict(
                    batch, horizon=horizon, quantile_levels=requested,
                    context_length=max(widths), **self.adjustments,
                )
            grid = out.quantiles.float().cpu().numpy().astype(float)  # [target rows, H, Q]
            for rows, flat in zip(chunk, target_rows, strict=True):
                for row, index in zip(rows, flat, strict=True):
                    point[row] = grid[index, :, median]
                    if quantiles is not None:
                        quantiles[row] = grid[index][:, [requested.index(q) for q in levels]]
        return AdapterOutput(point=point, quantiles=quantiles)

    def _series(self, panel: TimeSeriesPanel, rows: tuple[int, ...], horizon: int) -> tuple[Any, int]:
        """One item as a T0 ``TimeSeries`` and the number of target rows it carries."""
        import torch

        from ..t0 import TimeSeries

        length = max(len(panel.values[r]) for r in rows)
        past = dict(panel.past_covariates[rows[0]])
        future = dict(panel.future_covariates[rows[0]])
        for block in (past, future):
            for name, values in block.items():
                if np.asarray(values).dtype.kind not in "biuf":
                    raise ConfigError(
                        f"T0 accepts numeric covariates only; {name!r} is {np.asarray(values).dtype}. "
                        "Encode it (one-hot or ordinal) first."
                    )
        missing = [k for k in future if k not in past]
        if missing:
            raise ConfigError(
                f"Known covariate(s) {missing} have no history; T0 needs each known covariate "
                "over the context as well as the horizon."
            )
        known = [k for k in past if k in future]
        only = [k for k in past if k not in future]
        context_rows = [_right_aligned(panel.values[r], length) for r in rows]
        context_rows += [_right_aligned(past[k], length) for k in only]
        context = torch.as_tensor(np.stack(context_rows)[None], device=self.device)
        future_block = None
        if known:
            blocks = []
            for k in known:
                ahead = np.asarray(future[k], dtype=np.float32)[:horizon]
                if len(ahead) < horizon:
                    raise ConfigError(f"Known covariate {k!r} covers {len(ahead)} future steps; {horizon} are needed.")
                blocks.append(np.concatenate([_right_aligned(past[k], length), ahead]))
            future_block = torch.as_tensor(np.stack(blocks)[None], device=self.device)
        if future_block is None:
            item = TimeSeries.from_array(context, horizon=horizon)
        else:
            item = TimeSeries.from_array(context, future_block)
        return item, len(context_rows)

    def embed(self, panel: TimeSeriesPanel) -> np.ndarray:
        """Mean of the transformer outputs over the observed context patches, one vector per row."""
        import torch

        from ..t0 import TimeSeries
        from ..t0.types import MaskType

        model = self._model
        ps = int(model.patch_size)
        captured: dict[str, Any] = {}
        handle = model.transformer.register_forward_hook(lambda _m, _i, out: captured.__setitem__("h", out))
        vectors = []
        try:
            with torch.inference_mode():
                for start in range(0, len(panel), self.batch_size):
                    chunk = panel.values[start : start + self.batch_size]
                    series = [
                        TimeSeries.from_array(
                            torch.as_tensor(np.asarray(v, dtype=np.float32)[None], device=self.device), horizon=ps
                        )
                        for v in chunk
                    ]
                    batch = TimeSeries.batch(series)
                    padded = model.patcher.pad(batch, batch.seq_len - ps)
                    model._single_pass(padded)
                    hidden = captured["h"].float()  # [rows, patches, dim]
                    mask = model.patcher.patch(padded.mask)
                    observed = (mask == MaskType.VALID).any(-1)[..., None].to(hidden.dtype)
                    pooled = (hidden * observed).sum(1) / observed.sum(1).clamp_min(1.0)
                    vectors.append(pooled.cpu().numpy())
        finally:
            handle.remove()
        return np.concatenate(vectors).astype(float)

    def finetune(self, panel: TimeSeriesPanel, spec: TrainingSpec) -> dict[str, Any]:
        limit = int(self._model.max_horizon)
        spec = self.training_horizon(spec, limit, reason=f"predicts {limit} steps in one pass")
        try:
            report = super().finetune(panel, spec)
        finally:
            for p in self._model.parameters():
                p.requires_grad = False
            self._model.eval()
        return {
            "objective": "pinball over the native quantiles in the scaler's normalised space",
            "trained_on": "univariate windows without covariates",
            **report,
        }

    def _network(self) -> Any:
        return self._model

    def _training_loss(self, torch: Any, context: np.ndarray, label: np.ndarray) -> Any:
        """Pinball loss in the scaler's normalised space, over one differentiable pass.

        The window is the rollout buffer upstream predicts from (``finetune``
        clamps the horizon to one pass), scaled by the model's own causal
        scaler. The quantiles are read before ``rescale_predictions`` and the
        targets are mapped into the same per-patch frame, so nothing in the
        loss goes through ``sinh``, which overflows to non-finite gradients
        once the decoder saturates. This is upstream's ``_predict_step`` read
        backwards.
        """
        from ..t0 import TimeSeries
        from ..t0.rollout import prepare_rollout_buffer
        from ..t0.types import round_up

        model = self._model
        ps = int(model.patch_size)
        horizon, width = label.shape[1], context.shape[1]
        series = TimeSeries.from_array(
            torch.as_tensor(np.asarray(context, dtype=np.float32), device=self.device), horizon=horizon
        )
        context_width, forecast_width = round_up(width, ps), round_up(horizon, ps)
        window = prepare_rollout_buffer(series, horizon, width, ps).time_slice(0, context_width + forecast_width)
        with torch.no_grad():
            scaled, loc_scale = model.scaler.scale_input(window)
        raw = model(scaled)  # [rows, patches, patch_size, Q], normalised space
        first, patches = context_width // ps - 1, forecast_width // ps
        pred = raw[:, first : first + patches].reshape(raw.shape[0], forecast_width, raw.shape[-1])[:, :horizon]

        with torch.no_grad():
            loc = loc_scale.loc[:, ps - 1 :: ps][:, first : first + patches]
            scale = loc_scale.scale[:, ps - 1 :: ps][:, first : first + patches]
            loc = loc.repeat_interleave(ps, dim=-1)[:, :horizon].double()
            scale = scale.repeat_interleave(ps, dim=-1)[:, :horizon].double()
            target = (torch.as_tensor(label, dtype=torch.float64, device=pred.device) - loc) / scale
            if model.scaler.use_arcsinh:
                target = target.asinh()
            target = target.float()

        levels = torch.tensor(self.native_levels)
        ones = torch.ones(pred.shape[0], device=pred.device)
        return _training.pinball(torch, pred.float().permute(0, 2, 1), target, levels, ones)

def _right_aligned(values: Any, length: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)[-length:]
    row = np.full(length, np.nan, dtype=np.float32)
    row[length - len(values) :] = values
    return row
