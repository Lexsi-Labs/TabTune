"""Toto 2.0 adapter: the only place TabTune talks to the Toto API.

Toto 2.0 (``Datadog/Toto-2.0-*``) is deterministic (no seed), quantile-native and
multivariate. ``forecast`` returns the nine trained deciles, the point forecast is
the median, and a level off the trained grid is refused rather than interpolated.
An item's target variates are forecast jointly via
:meth:`~tabtune.TimeSeries.schema.TimeSeriesPanel.item_groups`.

The context must be a multiple of the checkpoint's patch size (upstream raises
otherwise), so histories are left-padded to a patch boundary with the padding
masked. Missing values are filled with upstream's ``ffill_imputation`` and
masked; the mask makes the fill unobservable. The backend needs the ``toto`` pip
extra. Toto has no covariate support (``supports_covariates=False``); to
condition on a related series, name it as an extra target.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

from ...registry.errors import ConfigError
from ._levels import exact_levels
from ._params import as_float, as_int, check_median_only, hub_kwargs, read_params
from .base import AdapterOutput, TrainingSpec, TSFMAdapter

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ...config.schemas import ForecastConfig
    from ...registry.TimeSeries import TimeSeriesModelSpec
    from ...TimeSeries.schema import TimeSeriesPanel

logger = logging.getLogger(__name__)

__all__ = ["Toto2Adapter"]

_KNOWN_KEYS = (
    "dtype",
    "revision",
    "local_files_only",
    "batch_size",
    "decode_block_size",
    "scaler_fallback_min_obs",
    "quantile_real_cap_k",
    "point_forecast",
)


class Toto2Adapter(TSFMAdapter):
    """Run Toto 2.0 (``Datadog/Toto-2.0-*``) behind the TSFM adapter interface.

    ``model_params`` accepted:
        batch_size: Items per forward pass (default 32). Memory grows with the
            context length and the variate count, so lower it for the 1B and
            2.5B checkpoints, which are published in float32.
        decode_block_size: Horizon steps decoded per pass (default 768, the value
            upstream's model card uses), which must be a multiple of the
            checkpoint's patch size. Blocks are decoded with a KV cache and
            median feedback between them, so this is **not** only a memory knob:
            a horizon longer than one block gives different numbers than one
            decoded in a single pass.
        scaler_fallback_min_obs: Backfill ``loc``/``scale`` on leading patches
            with fewer than this many observations (default 8, upstream's
            standard setting). ``0`` disables it.
        quantile_real_cap_k: Clip each quantile to the observed context range
            widened by this many scales (default 1e4, upstream's standard
            setting). ``0`` disables it.
        point_forecast: Only ``"median"``. Toto predicts quantiles directly and
            produces no sample paths, so there is no mean to take.

    No ``dtype``: every published checkpoint is float32, and casting a unit-scaled
    model breaks its variance assumptions. Forecasts are deterministic;
    ``tuning_params={"seed": ...}`` has no effect here.
    """

    point_forecast = "median"
    default_train_context = 1024
    lora_targets = ("attn.in_proj", "attn.out_proj", "ffn.fc1", "ffn.fc2")

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
        self.decode_block_size = as_int(
            params, "decode_block_size", 768, name=spec.name, minimum=1
        )
        self._resolve_dtype()
        self.scaler_fallback_min_obs = as_int(
            params, "scaler_fallback_min_obs", 8, name=spec.name, minimum=0
        )
        self.quantile_real_cap_k = as_float(
            params, "quantile_real_cap_k", 1e4, name=spec.name, minimum=0.0
        )

        check_median_only(params, name=spec.name)

    def load(self) -> None:
        try:
            from ..toto.toto2 import Toto2Model
        except ImportError as exc:  # the u-microP libraries are an optional extra
            raise self.missing_extra_error(exc, "dd_unit_scaling") from exc

        logger.info("[Toto2] Loading %s on %s", self.checkpoint, self.device)
        model = Toto2Model.from_pretrained(self.checkpoint, **hub_kwargs(self.model_params))
        model.to(self.device)
        self._restore_tuned(model)
        model.eval()
        self._model = model

        patch = model.config.patch_size
        if self.decode_block_size % patch != 0:
            raise ConfigError(
                f"Toto2 decode_block_size ({self.decode_block_size}) must be a multiple "
                f"of the checkpoint's patch size ({patch})."
            )

    def forecast(self, panel: TimeSeriesPanel, config: ForecastConfig) -> AdapterOutput:
        import torch

        horizon = config.prediction_length
        levels = [float(q) for q in config.quantile_levels]
        rows_for_level = self._quantile_rows(levels)
        median_row = self._trained_levels().index(0.5)

        groups = panel.item_groups()
        point = np.empty((len(panel), horizon), dtype="float64")
        quantiles_np = (
            np.empty((len(panel), horizon, len(levels)), dtype="float64") if levels else None
        )

        for start in range(0, len(groups), self.batch_size):
            chunk = groups[start : start + self.batch_size]
            values, mask = self._batch_for(panel, chunk)
            inputs = {
                "target": torch.as_tensor(values, dtype=torch.float32, device=self.device),
                "target_mask": torch.as_tensor(mask, dtype=torch.bool, device=self.device),
                "series_ids": torch.zeros(
                    values.shape[:2], dtype=torch.long, device=self.device
                ),
            }
            with torch.inference_mode():
                # [quantile, batch, variate, horizon]
                output = self._model.forecast(
                    inputs,
                    horizon,
                    decode_block_size=self.decode_block_size,
                    has_missing_values=not bool(mask.all()),
                    scaler_fallback_min_obs=self.scaler_fallback_min_obs,
                    quantile_real_cap_k=self.quantile_real_cap_k,
                )
            batch_quantiles = output.float().cpu().numpy().astype("float64")

            for entry, item_rows in enumerate(chunk):
                for variate, row in enumerate(item_rows):
                    point[row] = batch_quantiles[median_row, entry, variate]
                    if quantiles_np is not None:
                        quantiles_np[row] = batch_quantiles[
                            rows_for_level, entry, variate
                        ].transpose()

        return AdapterOutput(point=point, quantiles=quantiles_np)

    def embed(self, panel: TimeSeriesPanel) -> np.ndarray:
        """Mean of the transformer states over the patches that hold the series.

        The input to ``output_head`` is captured with a forward pre-hook. One
        state per patch; patches with no observation are excluded. Dimension is
        ``d_model``.
        """
        import torch

        net = self._model
        patch = int(net.config.patch_size)
        groups = panel.item_groups()
        vectors: dict[int, np.ndarray] = {}
        for start in range(0, len(groups), self.batch_size):
            chunk = groups[start : start + self.batch_size]
            values, mask = self._batch_for(panel, chunk)
            target = torch.as_tensor(values, dtype=torch.float32, device=self.device)
            observed = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
            captured: dict[str, Any] = {}
            handle = net.output_head.register_forward_pre_hook(
                lambda _module, args, store=captured: store.setdefault("x", args[0])
            )
            try:
                with torch.inference_mode():
                    net(
                        target=target,
                        target_mask=observed,
                        cpm_mask=torch.ones_like(observed),
                        series_ids=torch.zeros(
                            values.shape[:2], dtype=torch.long, device=self.device
                        ),
                    )
            finally:
                handle.remove()
            # [batch, variate, patch, d_model], and the per-patch observedness beside it.
            hidden = captured["x"].float()
            weights = observed.unflatten(-1, (-1, patch)).any(dim=-1)[..., None].to(hidden.dtype)
            pooled = (hidden * weights).sum(dim=-2) / weights.sum(dim=-2).clamp_min(1.0)
            pooled = pooled.cpu().numpy()
            for entry, item_rows in enumerate(chunk):
                for variate, row in enumerate(item_rows):
                    vectors[row] = pooled[entry, variate]
        return np.stack([vectors[row] for row in range(len(panel))]).astype(float)

    def finetune(self, panel: TimeSeriesPanel, spec: TrainingSpec) -> dict[str, Any]:
        net = self._model
        if int(net.config.num_output_patches) != 1:
            raise ConfigError(
                f"Fine-tuning Toto2 expects a checkpoint that predicts one patch per "
                f"position, but {self.checkpoint!r} predicts "
                f"{net.config.num_output_patches}. The next-patch alignment the training "
                f"loss relies on does not hold for such a checkpoint."
            )
        try:
            report = super().finetune(panel, spec)
        finally:
            for p in net.parameters():
                p.requires_grad = False
            net.eval()
        return {"objective": "next-patch pinball over the quantile knots", **report}

    def _network(self) -> Any:
        return self._model

    def _training_context(self, spec: TrainingSpec) -> int:
        patch = int(self._model.config.patch_size)
        limit = self.spec.max_context or self.default_train_context
        context = min(spec.context_length or self.default_train_context, limit)
        return max(patch, context - context % patch)

    def _training_loss(self, torch: Any, context: np.ndarray, label: np.ndarray) -> Any:
        """Toto's own objective: the state at each patch scores the patch after it.

        Targets are the ``asinh`` of the series scaled with the forward pass's
        ``loc``/``scale``; no ``context_scale`` division applies on top.
        ``cpm_mask`` must be a real tensor, not ``None``.
        """
        from ._training import pinball

        net = self._model
        patch = int(net.config.patch_size)
        window = np.concatenate([context, label], axis=1)
        pad = -window.shape[1] % patch
        if pad:
            window = np.concatenate(
                [np.full((len(window), pad), np.nan, dtype=np.float32), window], axis=1
            )
        observed = np.isfinite(window)

        values = torch.as_tensor(np.nan_to_num(window), dtype=torch.float32, device=self.device)
        values = values[:, None, :]
        mask = torch.as_tensor(observed, device=self.device)[:, None, :]
        output = net(
            target=values,
            target_mask=mask,
            cpm_mask=torch.ones_like(mask),
            series_ids=torch.zeros(values.shape[:2], dtype=torch.long, device=self.device),
        )
        scaled = torch.where(
            mask, (values - output.loc) / output.scale, torch.zeros_like(values)
        ).asinh()
        target = scaled.unflatten(-1, (-1, patch))[..., 1:, :]
        target_mask = mask.unflatten(-1, (-1, patch))[..., 1:, :]
        target = torch.where(target_mask, target, torch.full_like(target, float("nan")))
        # [quantile, batch, variate, patch, step] -> pinball's [batch, quantile, step]
        pred = output.quantiles[..., :-1, :]
        levels = pred.shape[0]
        pred = pred.permute(1, 2, 3, 0, 4).reshape(-1, levels, patch)
        target = target.reshape(-1, patch)
        return pinball(
            torch,
            pred,
            target,
            torch.tensor(self._trained_levels()),
            torch.ones(target.shape[0], device=self.device),
        )

    def quantile_range(self) -> tuple[float, float]:
        trained = self._trained_levels()
        return min(trained), max(trained)

    def _trained_levels(self) -> list[float]:
        """The quantile levels the loaded checkpoint actually predicts."""
        return [float(q) for q in self._model.output_head.knots]

    def _quantile_rows(self, levels: list[float]) -> list[int]:
        """Upstream's quantile rows for ``levels``, refusing any it never predicts."""
        return exact_levels(self._trained_levels(), levels, model=self.spec.name)

    def _batch_for(
        self, panel: TimeSeriesPanel, groups: tuple[tuple[int, ...], ...]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build ``(values, mask)`` of shape ``[item, variate, context]`` for one batch.

        Histories are left-padded to the batch's longest, rounded up to a patch
        boundary (upstream raises if the length does not divide), with the
        padding masked.
        """
        patch = self._model.config.patch_size
        longest = max(len(panel.values[rows[0]]) for rows in groups)
        context = max(math.ceil(longest / patch) * patch, patch)
        variates = len(groups[0])

        values = np.zeros((len(groups), variates, context), dtype="float64")
        mask = np.zeros((len(groups), variates, context), dtype=bool)
        for entry, item_rows in enumerate(groups):
            for variate, row in enumerate(item_rows):
                history = panel.values[row]
                observed = ~np.isnan(history)
                values[entry, variate, context - len(history) :] = self._impute(history)
                mask[entry, variate, context - len(history) :] = observed
        return values, mask

    def _impute(self, history: np.ndarray) -> np.ndarray:
        """Fill missing values with upstream's ``ffill_imputation``.

        The fill must be finite (NaN propagates through the masked arithmetic);
        its value is unobservable in the output.
        """
        if not np.isnan(history).any():
            return history

        from ..toto.toto2 import ffill_imputation

        return ffill_imputation(history.astype("float64"))
