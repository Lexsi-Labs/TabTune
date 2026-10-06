"""Time-MoE adapter (model code vendored in :mod:`tabtune.models.time_moe`).

Time-MoE (Shi et al., ICLR 2025) is a family of decoder-only univariate point
forecasters with sparse mixture-of-experts layers and several output heads,
each predicting the next ``h`` steps. The adapter z-normalises the context
(scale 1 for constant series, where upstream divides by zero), decodes in a
loop over ``forward`` using the largest head that fits the remaining horizon,
and de-normalises. Ragged batches are left-padded with an attention mask and
position ids counted from each series' first observation.

The context is cut so that context plus horizon fits ``max_position_embeddings``
(4096), but never below 512 steps. Fine-tuning uses the pretraining objective:
the Huber loss over every position and head, plus the router load-balancing term.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

from ...registry.errors import ConfigError
from ._params import as_int, hub_kwargs, read_params
from .base import AdapterOutput, TrainingSpec, TSFMAdapter

if TYPE_CHECKING:
    from ...config.schemas import ForecastConfig
    from ...registry.TimeSeries import TimeSeriesModelSpec
    from ...TimeSeries.schema import TimeSeriesPanel

logger = logging.getLogger(__name__)

__all__ = ["TimeMoEAdapter", "normalise_context"]

_KNOWN_KEYS = ("batch_size", "dtype", "attn_implementation", "revision", "local_files_only")
_DTYPES = ("float32", "bfloat16")
_MIN_CONTEXT = 512


def normalise_context(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Z-normalise a 1-D context; returns ``(normalised, mean, scale)``.

    The scale is 1 when the standard deviation is zero, non-finite or below
    ``1e-7`` times the mean, so a constant series is shifted, not divided by
    rounding noise.
    """
    values = np.asarray(values, dtype=np.float64)
    observed = values[np.isfinite(values)]
    mean = float(observed.mean()) if observed.size else 0.0
    std = float(observed.std(ddof=1)) if observed.size > 1 else 0.0
    if not np.isfinite(std) or std <= 0.0 or std <= 1e-7 * abs(mean):
        std = 1.0
    return (values - mean) / std, mean, std


class TimeMoEAdapter(TSFMAdapter):
    """Run Time-MoE (``Maple728/TimeMoE-50M``, ``-200M``) behind the adapter interface.

    ``model_params`` accepted:
        batch_size: Series per forward pass (default 64).
        dtype: ``"float32"`` (default) or ``"bfloat16"``.
        attn_implementation: ``"eager"`` (default) or ``"flash_attention_2"``
            (needs the ``flash-attn`` package and a CUDA GPU).
        revision, local_files_only: Hugging Face download options.

    Forecasts are point forecasts; :meth:`TimeSeriesPipeline.calibrate` adds
    conformal quantiles.
    """

    point_forecast = "mean"
    supported_dtypes = _DTYPES
    lora_targets = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj")

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
        self.batch_size = as_int(params, "batch_size", 64, name=spec.name, minimum=1)
        self.dtype = self._resolve_dtype()
        self.attn_implementation = params.get("attn_implementation", "eager")
        if self.attn_implementation not in ("eager", "flash_attention_2"):
            raise ConfigError(
                "TimeMoE attn_implementation must be 'eager' or 'flash_attention_2', "
                f"got {self.attn_implementation!r}"
            )

    def load(self) -> None:
        import torch

        from ..time_moe import TimeMoeConfig, TimeMoeForPrediction

        hub = hub_kwargs(self.model_params)
        config = TimeMoeConfig.from_pretrained(self.checkpoint, **hub)
        if getattr(config, "model_type", None) != "time_moe":
            raise ConfigError(
                f"{self.checkpoint!r} is not a Time-MoE checkpoint "
                f"(model_type={getattr(config, 'model_type', None)!r})"
            )
        config._attn_implementation = self.attn_implementation
        logger.info("[TimeMoE] Loading %s (%s) on %s", self.checkpoint, self.dtype, self.device)
        model = TimeMoeForPrediction.from_pretrained(
            self.checkpoint, config=config, dtype=torch.float32, **hub
        )
        model.to(self.device)
        self._restore_tuned(model)
        model.to(dtype=getattr(torch, self.dtype))
        model.eval()
        self._model = model

    @property
    def max_positions(self) -> int:
        return int(getattr(self._model.config, "max_position_embeddings", 4096) or 4096)

    def forecast(self, panel: TimeSeriesPanel, config: ForecastConfig) -> AdapterOutput:
        import torch

        horizon = int(config.prediction_length)
        cap = max(self.max_positions - horizon, min(_MIN_CONTEXT, self.max_positions // 2), 1)
        point = np.empty((len(panel), horizon))
        with torch.inference_mode():
            for start in range(0, len(panel), self.batch_size):
                chunk = panel.values[start : start + self.batch_size]
                stats = [normalise_context(np.asarray(v, dtype=float)[-cap:]) for v in chunk]
                x, mask = self._batch(torch, [s[0] for s in stats])
                pred = self._decode(torch, x, mask, horizon).cpu().numpy().astype(float)
                for i, (_, mean, scale) in enumerate(stats):
                    point[start + i] = pred[i] * scale + mean
        return AdapterOutput(point=point, quantiles=None)

    def embed(self, panel: TimeSeriesPanel) -> np.ndarray:
        """Mean of the decoder's final hidden states over the observed steps."""
        import torch

        cap = self.max_positions
        vectors = []
        dtype = next(self._model.parameters()).dtype
        with torch.inference_mode():
            for start in range(0, len(panel), self.batch_size):
                chunk = [np.asarray(v, dtype=float)[-cap:] for v in panel.values[start : start + self.batch_size]]
                x, mask = self._batch(torch, [normalise_context(c)[0] for c in chunk])
                hidden = self._model.model(
                    input_ids=x[..., None].to(dtype),
                    attention_mask=mask,
                    position_ids=_positions(torch, mask),
                    use_cache=False,
                    return_dict=True,
                ).last_hidden_state.float()
                weights = mask[..., None].to(hidden.dtype)
                pooled = (hidden * weights).sum(1) / weights.sum(1).clamp_min(1.0)
                vectors.append(pooled.cpu().numpy())
        return np.concatenate(vectors).astype(float)

    def finetune(self, panel: TimeSeriesPanel, spec: TrainingSpec) -> dict[str, Any]:
        import torch

        self._model.float()
        try:
            report = super().finetune(panel, spec)
        finally:
            for p in self._model.parameters():
                p.requires_grad = False
            self._model.to(dtype=getattr(torch, self.dtype))
        return {"objective": "huber over all heads + router load balancing", **report}

    def _network(self) -> Any:
        return self._model

    def _training_context(self, spec: TrainingSpec) -> int:
        limit = self.max_positions - spec.prediction_length - 1
        return max(1, min(spec.context_length or _MIN_CONTEXT, limit))

    def _training_loss(self, torch: Any, context: np.ndarray, label: np.ndarray) -> Any:
        window = np.concatenate([context, label], axis=1).astype(np.float64)
        normed = np.empty_like(window)
        for i in range(len(window)):
            _, mean, scale = normalise_context(context[i])
            normed[i] = (window[i] - mean) / scale
        inputs, labels = normed[:, :-1], normed[:, 1:]
        observed = np.isfinite(inputs)
        loss_mask = (np.isfinite(labels) & observed).astype(np.float32)
        device = self.device
        mask = torch.as_tensor(observed.astype(np.int64), device=device)
        out = self._model(
            input_ids=torch.as_tensor(np.nan_to_num(inputs), dtype=torch.float32, device=device)[..., None],
            attention_mask=mask,
            position_ids=_positions(torch, mask),
            labels=torch.as_tensor(np.nan_to_num(labels), dtype=torch.float32, device=device),
            loss_masks=torch.as_tensor(loss_mask, device=device),
            use_cache=False,
            return_dict=True,
        )
        return out.loss

    def _batch(self, torch: Any, contexts: list[np.ndarray]) -> tuple[Any, Any]:
        """Left-pad normalised contexts into ``[batch, time]`` values and an attention mask."""
        width = max(len(c) for c in contexts)
        x = np.zeros((len(contexts), width), dtype=np.float32)
        mask = np.zeros((len(contexts), width), dtype=np.int64)
        for i, c in enumerate(contexts):
            x[i, width - len(c) :] = np.nan_to_num(c)
            mask[i, width - len(c) :] = np.isfinite(c)
        return torch.as_tensor(x, device=self.device), torch.as_tensor(mask, device=self.device)

    def _decode(self, torch: Any, x: Any, mask: Any, horizon: int) -> Any:
        model = self._model
        size = int(model.config.input_size)
        dtype = next(model.parameters()).dtype
        steps = []
        remaining = horizon
        while remaining > 0:
            out = model(
                input_ids=x[..., None].to(dtype),
                attention_mask=mask,
                position_ids=_positions(torch, mask),
                use_cache=False,
                return_dict=True,
                max_horizon_length=remaining,
            )
            step = out.logits[:, -1, :].float().reshape(x.shape[0], -1, size)[..., 0][:, :remaining]
            steps.append(step)
            x = torch.cat([x, step.to(x.dtype)], dim=-1)
            mask = torch.cat([mask, torch.ones_like(step, dtype=mask.dtype)], dim=-1)
            remaining -= step.shape[1]
        return torch.cat(steps, dim=-1)[:, :horizon]


def _positions(torch: Any, mask: Any) -> Any:
    """Position ids counted from each row's first observed step (padding gets 1, as upstream)."""
    positions = mask.long().cumsum(-1) - 1
    return positions.masked_fill(mask == 0, 1)
