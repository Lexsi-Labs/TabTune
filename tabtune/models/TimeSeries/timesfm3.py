"""TimesFM 3.0 adapter (model code vendored in :mod:`tabtune.models.timesfm3`).

One forward pass sees an item's target variates together with its covariates,
concatenated along the variate axis. Forecasts are deterministic; the point
forecast is the median; the quantiles are the fixed grid 0.1 to 0.9 and levels
off that grid are refused rather than interpolated.

Covariates reach upstream as float arrays, one channel per covariate in
:attr:`~tabtune.TimeSeries.schema.TimeSeriesPanel.covariate_names` order. A
known covariate is one array spanning history and horizon; a past-only
covariate spans the history alone.

Fine-tuning minimises the pinball loss over the native quantiles through the
model's single-pass decoder. The patch normalisation has no finite gradient for
a patch with fewer than two distinct observations, so each training window is
cut to the whole patches it observes and windows with a constant patch are
skipped.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

from ...registry.errors import ConfigError
from ._levels import exact_levels
from ._params import as_bool, as_int, hub_kwargs, read_params
from .base import AdapterOutput, TrainingSpec, TSFMAdapter

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ...config.schemas import ForecastConfig
    from ...registry.TimeSeries import TimeSeriesModelSpec
    from ...TimeSeries.schema import TimeSeriesPanel

logger = logging.getLogger(__name__)

__all__ = ["TimesFM3Adapter"]

_KNOWN_KEYS = (
    "batch_size",
    "dtype",
    "revision",
    "local_files_only",
    "use_znorm",
    "make_positive",
    "use_symmetric_averaging",
    "padding_mode",
    "point_forecast",
)
_PADDING_MODES = ("none", "edge")


class TimesFM3Adapter(TSFMAdapter):
    """Run TimesFM 3.0 (``google/timesfm-3.0-pytorch``) behind the TSFM adapter interface.

    ``model_params`` accepted:
        batch_size: Items per forward pass (default 32), passed upstream as
            ``per_core_batch_size``. Upstream defaults to 4; 32 is a better fit
            for whole-panel forecasting, and memory grows with the context
            length and the variate count, so lower it for long histories.
        use_znorm: Standardise each series before forecasting and undo it
            afterwards. Off by default, as upstream.
        make_positive: Clip the forecast at zero for series whose history is
            non-negative. Off by default, as upstream.
        use_symmetric_averaging: Average the forecast with the negated-input
            forecast, which costs a second forward pass per item. Off by
            default, as upstream.
        padding_mode: ``"none"`` (default) or ``"edge"``, which repeats the last
            known-future covariate value out to the patch boundary.
        point_forecast: Only ``"median"``. TimesFM produces no sample paths, so
            there is no mean to take.

    The published checkpoint is float32 and ``dtype`` is not applied.
    Forecasts are deterministic; ``tuning_params={"seed": ...}`` only seeds
    fine-tuning.
    """

    point_forecast = "median"
    lora_targets = ("query_proj", "key_proj", "value_proj", "out_proj")

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
        self.use_znorm = as_bool(params, "use_znorm", False, name=spec.name)
        self.make_positive = as_bool(params, "make_positive", False, name=spec.name)
        self.use_symmetric_averaging = as_bool(
            params, "use_symmetric_averaging", False, name=spec.name
        )

        self.padding_mode = str(params.get("padding_mode", "none"))
        if self.padding_mode not in _PADDING_MODES:
            raise ConfigError(
                f"TimesFM3 padding_mode must be one of {_PADDING_MODES}, "
                f"got {self.padding_mode!r}"
            )

        requested_point = params.get("point_forecast", "median")
        if requested_point != "median":
            raise ConfigError(
                f"TimesFM3 only supports point_forecast='median', got "
                f"{requested_point!r}: it predicts quantiles directly and produces no "
                f"sample paths to average. Use the Chronos model for a mean forecast."
            )

    def load(self) -> None:
        from ..timesfm3 import TimesFM3Forecaster

        logger.info("[TimesFM3] Loading %s on %s", self.checkpoint, self.device)
        self._model = TimesFM3Forecaster.from_pretrained(
            self.checkpoint, device=self.device, **hub_kwargs(self.model_params)
        )
        self._restore_tuned(self._model.model)
        self._model.model.eval()

    def forecast(self, panel: TimeSeriesPanel, config: ForecastConfig) -> AdapterOutput:
        import torch

        horizon = config.prediction_length
        levels = [float(q) for q in config.quantile_levels]
        columns = self._quantile_columns(levels)

        groups = panel.item_groups()
        contexts = [self._target_for(panel, rows) for rows in groups]
        past_only = [self._past_only_channels(panel, rows) for rows in groups]
        past_future = [self._past_future_channels(panel, rows, horizon) for rows in groups]

        with torch.inference_mode():
            outputs = list(
                self._model.predict_batch(
                    contexts=contexts,
                    horizon=horizon,
                    past_only_covariates=past_only if past_only[0] is not None else None,
                    past_future_covariates=past_future if past_future[0] is not None else None,
                    return_quantiles=bool(levels),
                    use_znorm=self.use_znorm,
                    make_positive=self.make_positive,
                    use_symmetric_averaging=self.use_symmetric_averaging,
                    padding_mode=self.padding_mode,
                )
            )

        point = np.empty((len(panel), horizon), dtype="float64")
        quantiles_np = (
            np.empty((len(panel), horizon, len(levels)), dtype="float64") if levels else None
        )
        trained = len(self._trained_levels())
        for rows, output in zip(groups, outputs, strict=True):
            variates = len(rows)
            item_point = np.asarray(output.forecast, dtype="float64").reshape(variates, horizon)
            if quantiles_np is not None:
                item_quantiles = np.asarray(output.quantiles, dtype="float64").reshape(
                    variates, horizon, trained
                )[:, :, columns]
            for variate, row in enumerate(rows):
                point[row] = item_point[variate]
                if quantiles_np is not None:
                    quantiles_np[row] = item_quantiles[variate]

        return AdapterOutput(point=point, quantiles=quantiles_np)

    def embed(self, panel: TimeSeriesPanel) -> np.ndarray:
        """Mean of the transformer stack's output over the context patches.

        Upstream exposes this tensor itself, through ``decode``'s documented
        ``return_aux_outputs`` contract, as ``"__call__:transformer_output"``: it
        is the stack's output immediately before ``output_head``, so nothing about
        the patching or the normalisation is re-derived here. ``decode`` lays the
        sequence out as ``num_context_patches`` context positions followed by the
        horizon positions, so only the leading context positions are pooled and
        the forecast queries are left out. Targets only, no covariate channels.
        Dimension is ``model_dims``.
        """
        import torch

        net = self._model.model
        patch = int(net.input_patch_len)
        groups = panel.item_groups()
        vectors: dict[int, np.ndarray] = {}
        with torch.inference_mode():
            for start in range(0, len(groups), self.batch_size):
                chunk = groups[start : start + self.batch_size]
                targets = [self._target_for(panel, rows) for rows in chunk]
                width = max(t.shape[-1] for t in targets)
                variates = targets[0].shape[0]
                values = np.zeros((len(chunk), variates, width), dtype=np.float32)
                observed = np.zeros((len(chunk), variates, width), dtype=bool)
                for i, item in enumerate(targets):
                    values[i, :, width - item.shape[-1] :] = np.nan_to_num(item)
                    observed[i, :, width - item.shape[-1] :] = np.isfinite(item)
                # decode's `mask` marks positions to ignore, so it is the negation.
                _, aux = net.decode(
                    target=torch.as_tensor(values, device=self.device),
                    horizon=patch,
                    mask=torch.as_tensor(~observed.any(axis=1), device=self.device),
                    return_aux_outputs=True,
                )
                hidden = aux["__call__:transformer_output"].float()
                patches = math.ceil(width / patch)
                hidden = hidden[:, :, :patches]
                # A patch counts when it holds an observation, per variate.
                padded = np.zeros((len(chunk), variates, patches * patch), dtype=bool)
                padded[..., padded.shape[-1] - width :] = observed
                weights = torch.as_tensor(
                    padded.reshape(len(chunk), variates, patches, patch).any(axis=-1),
                    device=self.device,
                )[..., None].to(hidden.dtype)
                pooled = (hidden * weights).sum(dim=2) / weights.sum(dim=2).clamp_min(1.0)
                pooled = pooled.cpu().numpy()
                for entry, item_rows in enumerate(chunk):
                    for variate, row in enumerate(item_rows):
                        vectors[row] = pooled[entry, variate]
        return np.stack([vectors[row] for row in range(len(panel))]).astype(float)

    def finetune(self, panel: TimeSeriesPanel, spec: TrainingSpec) -> dict[str, Any]:
        net = self._model.model
        try:
            report = super().finetune(panel, spec)
        finally:
            for p in net.parameters():
                p.requires_grad = False
            net.eval()
        return {"objective": "pinball over the native quantiles", **report}

    def _network(self) -> Any:
        return self._model.model

    def _training_context(self, spec: TrainingSpec) -> int:
        patch = int(self._model.model.input_patch_len)
        limit = self.spec.max_context or self.default_train_context
        context = min(spec.context_length or self.default_train_context, limit)
        return max(patch, context - context % patch)

    def _training_loss(self, torch: Any, context: np.ndarray, label: np.ndarray) -> Any:
        from ._training import context_scale, pinball

        net = self._model.model
        patch = int(net.input_patch_len)
        levels = torch.tensor(self._trained_levels())
        scale = torch.as_tensor(context_scale(context), device=self.device)
        target = torch.as_tensor(label, dtype=torch.float32, device=self.device)
        rows = _usable_rows(context, patch)
        if not rows:
            raise ConfigError(
                "TimesFM3 fine-tuning needs windows with at least one whole patch "
                f"({patch} steps) of non-constant history; none of this batch qualifies."
            )
        total, count = 0.0, 0
        for width, members in rows.items():
            x = torch.as_tensor(_interpolated(context[members, -width:]), device=self.device)
            pred = net.decode.__wrapped__(net, target=x[:, None, :], horizon=label.shape[1])
            pred = pred[:, 0].permute(0, 2, 1)
            total = total + pinball(torch, pred, target[members], levels, scale[members]) * len(members)
            count += len(members)
        return total / count

    def quantile_range(self) -> tuple[float, float]:
        trained = self._trained_levels()
        return min(trained), max(trained)

    def _trained_levels(self) -> list[float]:
        """The quantile levels the loaded checkpoint actually predicts."""
        return [float(q) for q in self._model.config.quantiles]

    def _quantile_columns(self, levels: list[float]) -> list[int]:
        """Upstream's columns for ``levels``, refusing any it never predicts."""
        return exact_levels(self._trained_levels(), levels, model=self.spec.name)

    def _target_for(self, panel: TimeSeriesPanel, rows: tuple[int, ...]) -> np.ndarray:
        """Stack one item's target variates.

        A single row stays 1-D, which is what upstream expects for a univariate
        item; several rows stack into ``(n_variates, history)``.
        """
        histories = [panel.values[row] for row in rows]
        return histories[0] if len(rows) == 1 else np.stack(histories)

    def _past_only_channels(
        self, panel: TimeSeriesPanel, rows: tuple[int, ...]
    ) -> np.ndarray | None:
        """Covariates observed over the history only, as ``(n_channels, history)``."""
        past = panel.past_covariates[rows[0]]
        future = panel.future_covariates[rows[0]]
        names = [name for name in panel.covariate_names if name not in future]
        if not names:
            return None
        return np.stack([past[name] for name in names])

    def _past_future_channels(
        self, panel: TimeSeriesPanel, rows: tuple[int, ...], horizon: int
    ) -> np.ndarray | None:
        """Known-future covariates as ``(n_channels, history + horizon)``.

        Upstream takes one array spanning both periods rather than a pair, and
        reads the horizon back off its width, so history and future are joined
        here. Covariates are shared by an item's rows, so the first row decides.
        """
        past = panel.past_covariates[rows[0]]
        future = panel.future_covariates[rows[0]]
        names = [name for name in panel.covariate_names if name in future]
        if not names:
            return None
        return np.stack(
            [np.concatenate([past[name], future[name][:horizon]]) for name in names]
        )


def _interpolated(rows: np.ndarray) -> np.ndarray:
    """Rows with interior ``NaN`` filled linearly (edges take the nearest value)."""
    out = np.array(rows, dtype=np.float32)
    for row in out:
        bad = ~np.isfinite(row)
        if bad.any():
            idx = np.arange(len(row))
            row[bad] = np.interp(idx[bad], idx[~bad], row[~bad])
    return out


def _usable_rows(context: np.ndarray, patch: int) -> dict[int, list[int]]:
    """Training rows grouped by the whole-patch width they observe.

    Leading ``NaN`` padding is dropped, the width is cut to a multiple of
    ``patch``, and rows with a constant patch are left out: the model's patch
    normalisation has no finite gradient there.
    """
    groups: dict[int, list[int]] = {}
    for i, row in enumerate(context):
        observed = np.flatnonzero(np.isfinite(row))
        if observed.size == 0:
            continue
        width = (len(row) - observed[0]) // patch * patch
        if width < patch:
            continue
        patches = _interpolated(row[None, -width:])[0].reshape(-1, patch)
        if (patches.std(axis=1) == 0).any():
            continue
        groups.setdefault(int(width), []).append(i)
    return groups
