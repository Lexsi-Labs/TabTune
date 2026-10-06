"""TiRex adapter (model code vendored in :mod:`tabtune.models.tirex`).

TiRex (Auer et al., 2025) is a univariate xLSTM forecaster that predicts
quantiles 0.1 to 0.9 in patches of 32 steps and reads missing values
natively. The code and weights are under the NXAI Community License, which
requires the notice "Built with technology from NXAI".

Fine-tuning feeds sampled windows through the model's own forecasting path
and minimises the pinball loss over its nine quantiles, scaled by each
window's standard deviation.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

from ._levels import select_levels
from ._params import as_bool, as_int, hub_kwargs, read_params
from .base import AdapterOutput, TrainingSpec, TSFMAdapter

if TYPE_CHECKING:
    from ...config.schemas import ForecastConfig
    from ...registry.TimeSeries import TimeSeriesModelSpec
    from ...TimeSeries.schema import TimeSeriesPanel

logger = logging.getLogger(__name__)

__all__ = ["TiRexAdapter"]

_KNOWN_KEYS = (
    "batch_size",
    "full_rollout",
    "dynamic_padding",
    "dtype",
    "revision",
    "local_files_only",
)


class TiRexAdapter(TSFMAdapter):
    """Run TiRex (``NX-AI/TiRex``, ``NX-AI/TiRex-1.1-gifteval``) behind the adapter interface.

    ``model_params`` accepted:
        batch_size: Series per forward pass (default 512).
        full_rollout: Predict every patch of the horizon in one forward pass
            instead of one pass per patch; faster for long horizons.
        dynamic_padding: Pad the context only to the next multiple of the
            patch size instead of the training context length; faster for
            short series.
    """

    point_forecast = "median"
    refuses_other_dtypes = True
    lora_targets = ("ffn.proj_up_gate", "ffn.proj_up", "ffn.proj_down")

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
        self.batch_size = as_int(params, "batch_size", 512, name=spec.name, minimum=1)
        self.full_rollout = as_bool(params, "full_rollout", False, name=spec.name)
        self.dynamic_padding = as_bool(params, "dynamic_padding", False, name=spec.name)

    def load(self) -> None:
        from ..tirex import load_model

        logger.info("[TiRex] Loading %s on %s", self.checkpoint, self.device)
        model = load_model(
            self.checkpoint, device=self.device, **hub_kwargs(self.model_params)
        )
        self._restore_tuned(model)
        model.eval()
        self._model = model

    def quantile_range(self) -> tuple[float, float]:
        return min(self.native_levels), max(self.native_levels)

    @property
    def native_levels(self) -> tuple[float, ...]:
        return tuple(float(q) for q in self._model.config.quantiles)

    def forecast(self, panel: TimeSeriesPanel, config: ForecastConfig) -> AdapterOutput:
        horizon = int(config.prediction_length)
        quantiles, median = self._model.forecast(
            [np.asarray(v, dtype=np.float32) for v in panel.values],
            prediction_length=horizon,
            batch_size=self.batch_size,
            full_rollout=self.full_rollout,
            dynamic_padding=self.dynamic_padding,
        )
        grid = quantiles.float().numpy().astype(float)[:, :horizon]
        levels = [float(q) for q in config.quantile_levels]
        return AdapterOutput(
            point=median.float().numpy().astype(float).reshape(len(panel), -1)[:, :horizon],
            quantiles=(
                select_levels(grid, self.native_levels, levels, model="TiRex")
                if levels
                else None
            ),
        )

    def embed(self, panel: TimeSeriesPanel) -> np.ndarray:
        """Upstream's context embedding: hidden states of every block, averaged
        over the patches that hold the series, L2-normalised per block,
        concatenated and layer-normalised."""
        import torch
        import torch.nn.functional as F

        model = self._model
        patch = int(model.config.input_patch_size)
        limit = int(model.config.train_ctx_len)
        vectors = []
        for start in range(0, len(panel), self.batch_size):
            chunk = [np.asarray(v, dtype=np.float32)[-limit:] for v in panel.values[start : start + self.batch_size]]
            width = max(len(c) for c in chunk)
            x = np.full((len(chunk), width), np.nan, dtype=np.float32)
            for i, c in enumerate(chunk):
                x[i, width - len(c) :] = c
            hidden = model._embed_context(torch.as_tensor(x)).float().cpu()
            for i, c in enumerate(chunk):
                tokens = -(-len(c) // patch)
                pooled = hidden[i, -tokens:].mean(dim=0)
                pooled = F.normalize(pooled, p=2, dim=-1).flatten()
                vectors.append(F.layer_norm(pooled, (pooled.shape[-1],)).numpy())
        return np.stack(vectors).astype(float)

    def finetune(self, panel: TimeSeriesPanel, spec: TrainingSpec) -> dict[str, Any]:
        from ..tirex.models.slstm.cell import sLSTMCellTorch

        upstream = sLSTMCellTorch.slstm_forward_pointwise
        sLSTMCellTorch.slstm_forward_pointwise = staticmethod(_safe_pointwise)
        try:
            report = super().finetune(panel, spec)
        finally:
            sLSTMCellTorch.slstm_forward_pointwise = staticmethod(upstream)
            for p in self._model.parameters():
                p.requires_grad = False
            self._model.eval()
        return {"objective": "pinball over the native quantiles", **report}

    def _network(self) -> Any:
        return self._model

    def _training_context(self, spec: TrainingSpec) -> int:
        limit = int(self._model.config.train_ctx_len)
        return min(spec.context_length or limit, limit)

    def _training_loss(self, torch: Any, context: np.ndarray, label: np.ndarray) -> Any:
        from ._training import context_scale, pinball

        model = self._model
        model.train()
        pred = model(
            torch.as_tensor(context, dtype=torch.float32, device=self.device),
            prediction_length=label.shape[1],
            full_rollout=self.full_rollout,
            dynamic_padding=self.dynamic_padding,
        )
        target = torch.as_tensor(label, dtype=torch.float32, device=self.device)
        scale = torch.as_tensor(context_scale(context), device=self.device)
        levels = torch.tensor(self.native_levels)
        return pinball(torch, pred, target, levels, scale)


def _safe_pointwise(Wx: Any, Ry: Any, b: Any, states: Any) -> list[Any]:
    """Upstream's sLSTM pointwise step with gradients that stay finite.

    Upstream caps the gates as ``minimum(exp(x), 1)``. On the first step
    ``exp(x)`` overflows to ``inf``, and backpropagating ``0 * inf`` turns every
    weight into NaN after one optimiser step. ``exp(min(x, 0))`` gives the
    same values, so it is swapped in only while fine-tuning.
    """
    import torch
    import torch.nn.functional as F

    y, c, n, m = states
    raw = Wx + Ry + b
    iraw, fraw, zraw, oraw = torch.unbind(raw.view(raw.shape[0], 4, -1), dim=1)
    logfplusm = m + F.logsigmoid(torch.clamp(fraw, max=15))
    mnew = torch.where(torch.all(n == 0.0), iraw, torch.max(iraw, logfplusm))
    ogate = torch.sigmoid(oraw)
    igate = torch.exp(torch.clamp(iraw - mnew, max=0.0))
    fgate = torch.exp(torch.clamp(logfplusm - mnew, max=0.0))
    zgate = torch.tanh(zraw)
    cnew = fgate * c + igate * zgate
    nnew = fgate * n + igate
    hnew = ogate * cnew / nnew
    return [hnew, cnew, nnew, mnew]
