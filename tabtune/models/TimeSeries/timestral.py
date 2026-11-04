"""Timestral adapter (model code vendored in :mod:`tabtune.models.timestral`).

Timestral (Mistral AI, 2026) is a causal patch transformer trained with
Decoupled Horizon Forecasting: separate prediction tokens read the causally
encoded context, and a monotone head emits 21 quantiles per step. An item's
targets are forecast jointly; past-only, known-future and categorical
covariates condition the targets without being forecast. Horizons longer than
one decode block are decoded block by block on the median.

Items are batched with their histories left-padded with ``NaN``, which the
model reads as missing. Categorical covariates are factorised to codes and
target-encoded upstream.

Fine-tuning trains univariate target windows through ``predict_block``, with
the pinball loss over the native quantiles taken in the model's normalised
space: targets are mapped into the frame each horizon patch is decoded in,
the forecaster's refine step read backwards.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

from ...registry.errors import ConfigError
from ._levels import select_levels
from ._params import as_bool, as_float, as_int, hub_kwargs, read_params
from . import _training
from .base import AdapterOutput, TrainingSpec, TSFMAdapter

if TYPE_CHECKING:
    from ...config.schemas import ForecastConfig
    from ...registry.TimeSeries import TimeSeriesModelSpec
    from ...TimeSeries.schema import TimeSeriesPanel

logger = logging.getLogger(__name__)

__all__ = ["TimestralAdapter"]

_KNOWN_KEYS = (
    "batch_size",
    "symmetric_averaging",
    "clip_nonnegative",
    "per_shot_horizon_patches",
    "quantile_real_cap_k",
    "revision",
    "local_files_only",
    "dtype",
)


class TimestralAdapter(TSFMAdapter):
    """Run Timestral (``mistralai/Timestral-5M`` to ``-209M``) behind the adapter interface.

    ``model_params`` accepted:
        batch_size: Items per forward pass (default 64).
        symmetric_averaging: Average with the mirrored forecast of the negated
            series (default ``True``, as upstream; two passes).
        clip_nonnegative: Clamp at zero the series whose observed context is
            non-negative (default ``True``, as upstream).
        per_shot_horizon_patches: Patches decoded per block (default 32,
            capped at the checkpoint's block size).
        quantile_real_cap_k: Clamp forecasts to the observed range widened by
            ``k`` times the last patch scale (default 1e4; 0 disables).
        revision, local_files_only: Hugging Face download options.

    Forecasts are deterministic; ``tuning_params={"seed": ...}`` has no effect here.
    """

    point_forecast = "median"
    refuses_other_dtypes = True
    default_train_context = 2048
    lora_targets = ("wQKV", "wO", "mlp.w12", "mlp.w3")

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
        self.symmetric_averaging = as_bool(params, "symmetric_averaging", True, name=spec.name)
        self.clip_nonnegative = as_bool(params, "clip_nonnegative", True, name=spec.name)
        self.per_shot_horizon_patches = as_int(params, "per_shot_horizon_patches", 32, name=spec.name, minimum=1)
        self.quantile_real_cap_k = as_float(params, "quantile_real_cap_k", 1e4, name=spec.name, minimum=0.0)

    def load(self) -> None:
        from ..timestral import TimestralForecaster
        from ..timestral.loading import load_model

        logger.info("[Timestral] Loading %s on %s", self.checkpoint, self.device)
        model = load_model(self.checkpoint, device="cpu", **hub_kwargs(self.model_params))
        self._restore_tuned(model)
        model.to(self.device)
        model.transformer.rope.rebuild_inv_freq()
        model.eval()
        self._model = TimestralForecaster(
            model,
            per_shot_horizon_patches=self.per_shot_horizon_patches,
            quantile_real_cap_k=self.quantile_real_cap_k,
        )

    @property
    def _net(self) -> Any:
        return self._model.model

    @property
    def native_levels(self) -> tuple[float, ...]:
        return tuple(round(float(q), 6) for q in self._model.quantile_levels)

    def quantile_range(self) -> tuple[float, float]:
        return min(self.native_levels), max(self.native_levels)

    def forecast(self, panel: TimeSeriesPanel, config: ForecastConfig) -> AdapterOutput:
        horizon = int(config.prediction_length)
        levels = [float(q) for q in config.quantile_levels]
        groups = panel.item_groups()
        point = np.empty((len(panel), horizon))
        quantiles = np.empty((len(panel), horizon, len(levels))) if levels else None
        for start in range(0, len(groups), self.batch_size):
            chunk = groups[start : start + self.batch_size]
            grid = self._forecast_items(panel, chunk, horizon)  # [items, targets, H, Q]
            for item, rows in enumerate(chunk):
                for j, row in enumerate(rows):
                    point[row] = select_levels(grid[item, j], self.native_levels, [0.5])[:, 0]
                    if quantiles is not None:
                        quantiles[row] = select_levels(grid[item, j], self.native_levels, levels, model="Timestral")
        return AdapterOutput(point=point, quantiles=quantiles)

    def _forecast_items(self, panel: TimeSeriesPanel, chunk: list[tuple[int, ...]], horizon: int) -> np.ndarray:
        import torch

        length = max(len(panel.values[r]) for rows in chunk for r in rows)
        n_targets = len(chunk[0])
        context = np.full((len(chunk), n_targets, length), np.nan, dtype=np.float32)
        for i, rows in enumerate(chunk):
            for j, r in enumerate(rows):
                values = np.asarray(panel.values[r], dtype=np.float32)
                context[i, j, length - len(values) :] = values
        covariates = _covariate_blocks(panel, chunk, length, horizon)
        if covariates.get("known_covariates") is not None or covariates.get("categorical_known_covariates") is not None:
            if not self._model.supports_known_covariates:
                raise ConfigError(f"{self.checkpoint} cannot condition on known-future covariates")
        with torch.inference_mode():
            out = self._model.forecast(
                context,
                prediction_length=horizon,
                symmetric_averaging=self.symmetric_averaging,
                clip_nonnegative=self.clip_nonnegative,
                **covariates,
            )
        return out.quantiles.float().cpu().numpy().astype(float)

    def embed(self, panel: TimeSeriesPanel) -> np.ndarray:
        """Mean of the final encoder states over the observed context patches, one vector per target."""
        import torch

        from ..timestral.forecaster import ffill_with_mask, pad_to_patch

        net = self._net
        ps = int(net.config.patch_size)
        captured: dict[str, Any] = {}
        handle = net.transformer.register_forward_hook(lambda _m, _i, out: captured.__setitem__("h", out))
        vectors = []
        try:
            with torch.inference_mode():
                for start in range(0, len(panel), self.batch_size):
                    chunk = panel.values[start : start + self.batch_size]
                    width = max(len(v) for v in chunk)
                    padded = np.full((len(chunk), width), np.nan, dtype=np.float32)
                    for i, series in enumerate(chunk):
                        padded[i, width - len(series) :] = np.asarray(series, dtype=np.float32)
                    x, mask = self._model._as_context(padded[:, None, :], None)
                    rows, mask = pad_to_patch(ffill_with_mask(x, mask)[:, 0], mask[:, 0], ps)
                    rows, mask = rows[:, -self._model.max_context :], mask[:, -self._model.max_context :]
                    group_ids = torch.arange(rows.shape[0], device=rows.device)
                    net.predict_block(rows, mask, group_ids, 1)
                    patches = rows.shape[-1] // ps
                    hidden = captured["h"][:, :patches].float()
                    observed = mask.unflatten(-1, (patches, ps)).any(-1)[..., None].to(hidden.dtype)
                    pooled = (hidden * observed).sum(1) / observed.sum(1).clamp_min(1.0)
                    vectors.append(pooled.cpu().numpy())
        finally:
            handle.remove()
        return np.concatenate(vectors).astype(float)

    def finetune(self, panel: TimeSeriesPanel, spec: TrainingSpec) -> dict[str, Any]:
        cfg = self._net.config
        limit = int(cfg.block_size) * int(cfg.patch_size)
        spec = self.training_horizon(spec, limit, reason=f"decodes {limit} steps per block")
        try:
            report = super().finetune(panel, spec)
        finally:
            for p in self._net.parameters():
                p.requires_grad = False
            self._net.eval()
        return {
            "objective": "pinball over the native quantiles in the normalised refine frame",
            "trained_on": "univariate windows without covariates",
            **report,
        }

    def _network(self) -> Any:
        return self._net

    def _training_context(self, spec: TrainingSpec) -> int:
        limit = int(self._net.config.context_length)
        return min(spec.context_length or self.default_train_context, limit)

    def _training_loss(self, torch: Any, context: np.ndarray, label: np.ndarray) -> Any:
        """Pinball loss in the model's normalised space, patch by patch in the refine frame.

        The head's quantiles are taken before denormalisation (a forward hook on
        ``head``), and the targets are mapped into the same space with the frame
        each horizon patch is decoded in: the last context patch's frame for the
        first patch, then the advancing frames of ``advancing_block_frames``
        built from the block's own median. The median is detached for the
        frames, so no gradient runs through their ``sqrt``, and nothing in the
        loss goes through ``sinh``. Both would overflow to non-finite gradients
        once the head saturates. With every target inside its frame, this is
        the forecaster's refine step read backwards.
        """
        from ..timestral.scaler import advancing_block_frames

        forecaster = self._model
        net = self._net
        cfg = net.config
        ps = int(cfg.patch_size)
        horizon = label.shape[1]
        patches = math.ceil(horizon / ps)
        x, mask = forecaster._as_context(np.asarray(context, dtype=np.float32)[:, None, :], None)
        rows, mask, _, _, _, _ = forecaster._assemble_rows(x, mask, horizon, None, None)
        rows, mask = rows[:, 0], mask[:, 0]
        group_ids = torch.arange(rows.shape[0], device=rows.device)

        captured: dict[str, Any] = {}
        handle = net.head.register_forward_hook(lambda _m, _i, out: captured.__setitem__("q", out))
        try:
            net.predict_block(rows, mask, group_ids, patches)
        finally:
            handle.remove()
        preds = captured["q"].float()  # [B, patches * ps, Q], normalised space

        with torch.no_grad():
            _, loc, scale = net.scale(rows, mask)
            loc0, scale0 = loc[:, -1].double(), scale[:, -1].double()
            locs = loc0[:, None].expand(-1, patches).clone()
            scales = scale0[:, None].expand(-1, patches).clone()
            if patches > 1:
                median = preds[:, :, forecaster.median_index].double().unflatten(-1, (patches, ps))
                sinh_median = median.clamp(-30.0, 30.0).sinh() if cfg.use_asinh else median
                frame_loc, frame_scale = advancing_block_frames(rows, mask, sinh_median, cfg.minimum_scale)
                locs[:, 1:], scales[:, 1:] = frame_loc[:, 1:], frame_scale[:, 1:]
            locs = locs.repeat_interleave(ps, dim=-1)[:, :horizon]
            scales = scales.repeat_interleave(ps, dim=-1)[:, :horizon]
            target = torch.as_tensor(label, dtype=torch.float64, device=preds.device)
            target = (target - locs) / scales
            if cfg.use_asinh:
                target = target.asinh()
            target = target.float()

        levels = torch.tensor(self.native_levels)
        ones = torch.ones(preds.shape[0], device=preds.device)
        return _training.pinball(torch, preds[:, :horizon].permute(0, 2, 1), target, levels, ones)

def _is_numeric(values: Any) -> bool:
    return np.asarray(values).dtype.kind in "biuf"


def _right_aligned(values: Any, length: int, dtype: Any = np.float32) -> np.ndarray:
    values = np.asarray(values)[-length:]
    row = np.full(length, np.nan, dtype=dtype)
    row[length - len(values) :] = values.astype(dtype)
    return row


def _codes(values: np.ndarray) -> tuple[np.ndarray, dict]:
    import pandas as pd

    codes, uniques = pd.factorize(pd.Series(values, dtype=object), use_na_sentinel=True)
    out = codes.astype(np.float64)
    out[codes < 0] = np.nan
    return out, {value: i for i, value in enumerate(uniques)}


def _covariate_blocks(
    panel: TimeSeriesPanel, chunk: list[tuple[int, ...]], length: int, horizon: int
) -> dict[str, np.ndarray | None]:
    """Past, known and categorical covariate arrays for a chunk of items."""
    past0 = dict(panel.past_covariates[chunk[0][0]])
    future0 = dict(panel.future_covariates[chunk[0][0]])
    missing = [k for k in future0 if k not in past0]
    if missing:
        raise ConfigError(
            f"Known covariate(s) {missing} have no history; Timestral needs each known "
            "covariate over the context as well as the horizon."
        )
    known = [k for k in past0 if k in future0]
    only = [k for k in past0 if k not in future0]
    blocks: dict[str, list[list[np.ndarray]]] = {
        "past_covariates": [], "known_covariates": [],
        "categorical_past_covariates": [], "categorical_known_covariates": [],
    }
    for rows in chunk:
        past = dict(panel.past_covariates[rows[0]])
        future = dict(panel.future_covariates[rows[0]])
        num_past, num_known, cat_past, cat_known = [], [], [], []
        for k in only:
            if _is_numeric(past[k]):
                num_past.append(_right_aligned(past[k], length))
            else:
                codes, _ = _codes(np.asarray(past[k], dtype=object))
                cat_past.append(_right_aligned(codes, length, np.float64))
        for k in known:
            ahead = np.asarray(future[k])[:horizon]
            if len(ahead) < horizon:
                raise ConfigError(f"Known covariate {k!r} covers {len(ahead)} future steps; {horizon} are needed.")
            if _is_numeric(past[k]) and _is_numeric(ahead):
                num_known.append(np.concatenate([_right_aligned(past[k], length), ahead.astype(np.float32)]))
            else:
                joined = np.concatenate([np.asarray(past[k], dtype=object)[-length:], ahead.astype(object)])
                codes, _ = _codes(joined)
                history = codes[: len(joined) - horizon]
                cat_known.append(np.concatenate([_right_aligned(history, length, np.float64), codes[-horizon:]]))
        blocks["past_covariates"].append(num_past)
        blocks["known_covariates"].append(num_known)
        blocks["categorical_past_covariates"].append(cat_past)
        blocks["categorical_known_covariates"].append(cat_known)
    out: dict[str, np.ndarray | None] = {}
    for name, per_item in blocks.items():
        out[name] = np.stack([np.stack(rows) for rows in per_item]) if per_item and per_item[0] else None
    return out
