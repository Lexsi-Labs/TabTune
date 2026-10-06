"""TiRex-2 adapter (model code vendored in :mod:`tabtune.models.tirex2`).

TiRex-2 (NX-AI, 2026) is a patch-based xLSTM forecaster with attention across
variates. It forecasts several targets of one item jointly and conditions on
past-only and known-future numeric covariates. It predicts quantiles 0.1 to
0.9 and reads missing values natively.

Each item is one input: its targets are forecast together and its covariates
go with them. Horizons longer than the checkpoint's ``future_len`` are
unrolled in ``future_len`` chunks, appending the median forecast to the
targets and ``NaN`` to past-only covariates.

Fine-tuning trains on univariate target windows through the model's own
prediction path (postprocessor included), with the pinball loss over its
quantiles scaled by each window's standard deviation.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

from ...registry.errors import ConfigError
from ._levels import select_levels
from ._params import as_int, hub_kwargs, read_params
from .base import AdapterOutput, TrainingSpec, TSFMAdapter

if TYPE_CHECKING:
    from ...config.schemas import ForecastConfig
    from ...registry.TimeSeries import TimeSeriesModelSpec
    from ...TimeSeries.schema import TimeSeriesPanel

logger = logging.getLogger(__name__)

__all__ = ["TiRex2Adapter"]

_KNOWN_KEYS = (
    "batch_size",
    "tta_sign_flip",
    "tta_diff",
    "pad_context",
    "compile",
    "revision",
    "local_files_only",
    "dtype",
)


class TiRex2Adapter(TSFMAdapter):
    """Run TiRex-2 (``NX-AI/TiRex-2`` and its variants) behind the adapter interface.

    ``model_params`` accepted:
        batch_size: Items per forward pass (default 256).
        tta_sign_flip: Also forecast the negated series and average (about
            twice the cost). ``None`` (default) keeps the checkpoint's setting.
        tta_diff: Difference trending targets in the postprocessor. ``None``
            (default) keeps the checkpoint's setting.
        pad_context: Pad short contexts to the full training length (default
            ``True``); ``False`` is faster and slightly less accurate.
        compile: ``torch.compile`` the recurrent and residual layers.
        revision, local_files_only: Hugging Face download options.
    """

    point_forecast = "median"
    refuses_other_dtypes = True
    lora_targets = (
        "attn.wq",
        "attn.wk",
        "attn.wv",
        "attn.wo",
        "ffn.proj_up",
        "ffn.proj_down",
        "ffn.wi",
        "ffn.wo",
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
        self._resolve_dtype()
        self.batch_size = as_int(params, "batch_size", 256, name=spec.name, minimum=1)

    def load(self) -> None:
        from ..tirex2 import load_model

        family = self.device.partition(":")[0]
        if family not in ("cpu", "cuda", "mps"):
            raise ConfigError(
                f"{self.spec.name} runs on cpu, cuda or mps, not {self.device!r}"
            )
        hub = hub_kwargs(self.model_params)
        logger.info("[TiRex2] Loading %s on %s", self.checkpoint, self.device)
        model = load_model(
            self.checkpoint,
            device=self.device,
            hf_kwargs=hub or None,
            compile=bool(self.model_params.get("compile", False)),
        )
        self._restore_tuned(model.model)
        model.model.eval()
        self._model = model

    @property
    def _net(self) -> Any:
        return self._model.model

    def quantile_range(self) -> tuple[float, float]:
        return min(self.native_levels), max(self.native_levels)

    @property
    def native_levels(self) -> tuple[float, ...]:
        return tuple(round(float(q), 6) for q in self._net.quantiles)

    def _predict_kwargs(self) -> dict[str, Any]:
        kwargs = {
            key: bool(self.model_params[key])
            for key in ("tta_sign_flip", "tta_diff", "pad_context")
            if self.model_params.get(key) is not None
        }
        return kwargs

    def forecast(self, panel: TimeSeriesPanel, config: ForecastConfig) -> AdapterOutput:
        horizon = int(config.prediction_length)
        levels = [float(q) for q in config.quantile_levels]
        groups = panel.item_groups()
        inputs = [self._input(panel, rows, horizon) for rows in groups]
        point = np.empty((len(panel), horizon))
        quantiles = np.empty((len(panel), horizon, len(levels))) if levels else None
        for rows, out in zip(groups, self._forecast_inputs(inputs, horizon), strict=True):
            grid = np.moveaxis(out[:, :, :horizon], 1, -1)
            for j, row in enumerate(rows):
                if quantiles is not None:
                    quantiles[row] = select_levels(
                        grid[j], self.native_levels, levels, model="TiRex2"
                    )
                point[row] = select_levels(grid[j], self.native_levels, [0.5])[:, 0]
        return AdapterOutput(point=point, quantiles=quantiles)

    def _input(self, panel: TimeSeriesPanel, rows: tuple[int, ...], horizon: int) -> Any:
        """One item's targets and covariates as a ``TimeseriesType``."""
        import torch

        from ..tirex2 import TimeseriesType

        length = max(len(panel.values[r]) for r in rows)
        target = np.full((len(rows), length), np.nan, dtype=np.float32)
        for j, r in enumerate(rows):
            values = np.asarray(panel.values[r], dtype=np.float32)
            target[j, length - len(values) :] = values
        past = dict(panel.past_covariates[rows[0]])
        future = dict(panel.future_covariates[rows[0]])
        _require_numeric(past, future)
        missing = [k for k in future if k not in past]
        if missing:
            raise ConfigError(
                f"Known covariate(s) {missing} have no history; TiRex-2 needs each known "
                "covariate over the context as well as the horizon."
            )
        known = [k for k in past if k in future]
        only = [k for k in past if k not in future]
        past_block = np.stack([_right_aligned(past[k], length) for k in only]) if only else None
        future_block = None
        if known:
            blocks = []
            for k in known:
                ahead = np.asarray(future[k], dtype=np.float32)[:horizon]
                if len(ahead) < horizon:
                    raise ConfigError(
                        f"Known covariate {k!r} covers {len(ahead)} future steps; {horizon} are needed."
                    )
                blocks.append(np.concatenate([_right_aligned(past[k], length), ahead]))
            future_block = np.stack(blocks)
        return TimeseriesType(
            target=torch.as_tensor(target),
            past_covariates=None if past_block is None else torch.as_tensor(past_block),
            future_covariates=None if future_block is None else torch.as_tensor(future_block),
        )

    def _forecast_inputs(self, inputs: list[Any], horizon: int) -> list[np.ndarray]:
        """``[targets, quantile, horizon]`` per input, unrolled past ``future_len``."""
        import torch

        from ..tirex2 import TimeseriesType

        step = int(self._net.future_len)
        kwargs = self._predict_kwargs()
        pieces: list[list[np.ndarray]] = []
        current = list(inputs)
        remaining = horizon
        while remaining > 0:
            h = min(step, remaining)
            out = self._model.forecast(
                current, prediction_length=h, output_type="numpy", batch_size=self.batch_size, **kwargs
            )
            out = [np.asarray(o, dtype=float) for o in out]
            pieces.append(out)
            remaining -= h
            if remaining <= 0:
                break
            extended = []
            for ts, arr in zip(current, out, strict=True):
                median = select_levels(np.moveaxis(arr, 1, -1), self.native_levels, [0.5])[..., 0]
                past = ts.past_covariates
                if past is not None:
                    past = torch.cat([past, torch.full((past.shape[0], h), float("nan"))], dim=-1)
                extended.append(
                    TimeseriesType(
                        target=torch.cat([ts.target, torch.as_tensor(median, dtype=ts.target.dtype)], dim=-1),
                        past_covariates=past,
                        future_covariates=ts.future_covariates,
                    )
                )
            current = extended
        return [np.concatenate([p[i] for p in pieces], axis=-1) for i in range(len(inputs))]

    def embed(self, panel: TimeSeriesPanel) -> np.ndarray:
        """Upstream's embedding: block-stack outputs averaged over the observed
        patches, one vector per target; an item's targets are encoded together."""
        import torch

        from ..tirex2 import TimeseriesType

        groups = panel.item_groups()
        inputs = []
        for rows in groups:
            length = max(len(panel.values[r]) for r in rows)
            target = np.full((len(rows), length), np.nan, dtype=np.float32)
            for j, r in enumerate(rows):
                values = np.asarray(panel.values[r], dtype=np.float32)
                target[j, length - len(values) :] = values
            inputs.append(TimeseriesType(torch.as_tensor(target), None, None))
        pad = self.model_params.get("pad_context")
        vectors = self._model.embed(
            inputs, batch_size=self.batch_size, pad_context=True if pad is None else bool(pad)
        )
        out = np.empty((len(panel), int(vectors[0].shape[-1])))
        for rows, emb in zip(groups, vectors, strict=True):
            out[list(rows)] = emb.float().numpy()
        return out

    def finetune(self, panel: TimeSeriesPanel, spec: TrainingSpec) -> dict[str, Any]:
        spec = self.training_horizon(
            spec,
            int(self._net.future_len),
            reason=f"predicts {int(self._net.future_len)} steps in one pass",
        )
        try:
            report = super().finetune(panel, spec)
        finally:
            for p in self._net.parameters():
                p.requires_grad = False
            self._net.eval()
        return {"objective": "pinball over the native quantiles", **report}

    def _network(self) -> Any:
        return self._net

    def _training_context(self, spec: TrainingSpec) -> int:
        limit = int(self._net.context_len)
        return min(spec.context_length or limit, limit)

    def _training_loss(self, torch: Any, context: np.ndarray, label: np.ndarray) -> Any:
        from ..tirex2 import TimeseriesType
        from ._training import context_scale, pinball

        net = self._net
        net.train()
        series = [TimeseriesType(torch.as_tensor(row)[None], None, None) for row in context]
        out = net.predict(series, label.shape[1], preserve_grad=True, **self._predict_kwargs())
        pred = torch.cat(out, dim=0)
        target = torch.as_tensor(label, dtype=torch.float32, device=pred.device)
        scale = torch.as_tensor(context_scale(context), device=pred.device)
        return pinball(torch, pred, target, torch.tensor(self.native_levels), scale)


def _right_aligned(values: Any, length: int) -> np.ndarray:
    """A covariate history in a ``length`` row, ``NaN`` before it starts."""
    values = np.asarray(values, dtype=np.float32)[-length:]
    row = np.full(length, np.nan, dtype=np.float32)
    row[length - len(values) :] = values
    return row


def _require_numeric(*blocks: Mapping[str, Any]) -> None:
    for block in blocks:
        for name, values in block.items():
            if np.asarray(values).dtype.kind not in "biuf":
                raise ConfigError(
                    f"TiRex-2 accepts numeric covariates only; {name!r} is "
                    f"{np.asarray(values).dtype}. Encode it (one-hot or ordinal) first."
                )
