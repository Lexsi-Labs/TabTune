"""Chronos v1 adapter: the only place TabTune talks to the Chronos API.

Chronos v1 tokenises each series, samples future token paths from a T5
encoder-decoder and decodes them back. ``ChronosPipeline.predict_quantiles``
takes a list of 1-D float tensors of any lengths (NaN is missing) and returns
``quantiles`` of shape ``[series, horizon, quantile]`` and ``mean`` of shape
``[series, horizon]``, float32 on CPU, in input order.

Horizons beyond the checkpoint's built-in 64 steps are produced autoregressively
by Chronos itself; the registry envelope warns about the quality loss.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np

from ..._internal.deprecation import warn_once
from ...registry.errors import ConfigError
from ._params import as_float, as_int, hub_kwargs, read_params
from .base import AdapterOutput, TrainingSpec, TSFMAdapter

if TYPE_CHECKING:
    from ...config.schemas import ForecastConfig
    from ...registry.TimeSeries import TimeSeriesModelSpec
    from ...TimeSeries.schema import TimeSeriesPanel

logger = logging.getLogger(__name__)

__all__ = ["ChronosAdapter"]

_SAMPLING_KEYS = ("num_samples", "temperature", "top_k", "top_p")
_KNOWN_KEYS = (*_SAMPLING_KEYS, "batch_size", "dtype", "point_forecast", "revision", "local_files_only")
_DTYPES = ("float32", "bfloat16")
_POINT_FORECASTS = ("mean", "median")


class ChronosAdapter(TSFMAdapter):
    """Run Chronos v1 (``amazon/chronos-t5-*``) behind the TSFM adapter interface.

    ``model_params`` accepted:
        num_samples, temperature, top_k, top_p: Sampling settings forwarded to
            Chronos. Omitted values use the checkpoint's defaults.
        batch_size: Series per forward pass (default 256, as upstream's
            ``predict_df``). Bounds memory on large panels.
        dtype: ``"float32"`` or ``"bfloat16"``. Defaults to bfloat16 on CUDA,
            where upstream recommends it, and float32 elsewhere.
        point_forecast: ``"mean"`` (default, what ``predict_quantiles``
            returns) or ``"median"`` (the 0.5 quantile of the samples).

    Forecasts are sampled, so pass ``tuning_params={"seed": ...}`` for
    reproducible output. Seeding uses a forked RNG, leaving the global torch
    random state untouched. A fixed seed reproduces a forecast only for a
    fixed ``batch_size``.
    """

    supported_dtypes = _DTYPES
    cuda_default_dtype = "bfloat16"
    respects_seed = True
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
        self.sampling_kwargs = {k: params[k] for k in _SAMPLING_KEYS if params.get(k) is not None}
        for key in ("temperature", "top_p"):
            if key in self.sampling_kwargs:
                self.sampling_kwargs[key] = as_float(params, key, 1.0, name=spec.name, minimum=0.0)
        for key in ("num_samples", "top_k"):
            if key in self.sampling_kwargs:
                self.sampling_kwargs[key] = as_int(params, key, 1, name=spec.name, minimum=1)
        self.batch_size = as_int(params, "batch_size", 256, name=spec.name, minimum=1)
        self.dtype = self._resolve_dtype()
        self.point_forecast = params.get("point_forecast", "mean")
        if self.point_forecast not in _POINT_FORECASTS:
            raise ConfigError(
                f"{spec.name} point_forecast must be one of {_POINT_FORECASTS}, "
                f"got {self.point_forecast!r}"
            )

    def load(self) -> None:
        import torch

        from ..chronos import ChronosPipeline

        logger.info("[Chronos] Loading %s (%s) on %s", self.checkpoint, self.dtype, self.device)
        # Load in float32: fine-tuned weights are saved as float32 and must not be rounded on restore.
        pipeline = ChronosPipeline.from_pretrained(
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

        extra_median = 0.5 not in levels and (self.point_forecast == "median" or not levels)
        requested = [*levels, 0.5] if extra_median else levels

        contexts = [torch.as_tensor(values, dtype=torch.float32) for values in panel.values]
        quantile_chunks: list[np.ndarray] = []
        mean_chunks: list[np.ndarray] = []

        with self._seeded_rng(torch), torch.inference_mode():
            for start in range(0, len(contexts), self.batch_size):
                quantiles, mean = self._model.predict_quantiles(
                    contexts[start : start + self.batch_size],
                    prediction_length=config.prediction_length,
                    quantile_levels=requested,
                    limit_prediction_length=False,
                    **self.sampling_kwargs,
                )
                quantile_chunks.append(quantiles.float().numpy())
                mean_chunks.append(mean.float().numpy())

        quantiles_np = np.concatenate(quantile_chunks, axis=0)
        if self.point_forecast == "median":
            point = quantiles_np[:, :, requested.index(0.5)]
        else:
            point = np.concatenate(mean_chunks, axis=0)
        if extra_median:
            quantiles_np = quantiles_np[:, :, :-1]

        return AdapterOutput(point=point, quantiles=quantiles_np if levels else None)

    def embed(self, panel: TimeSeriesPanel) -> np.ndarray:
        """Mean of the encoder states over the tokens that hold the series.

        Padding tokens and the trailing EOS token are excluded. Dimension is
        ``d_model``.
        """
        import torch

        tokenizer = self._model.tokenizer
        limit = int(tokenizer.config.context_length)
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
                token_ids, mask, _ = tokenizer.context_input_transform(
                    torch.as_tensor(padded)
                )
                hidden = self._model.model.encode(
                    input_ids=token_ids.to(self.device),
                    attention_mask=mask.to(self.device),
                ).float()
                # context_input_transform appends an EOS token, which is not an observation.
                observed = mask.to(self.device)
                if tokenizer.config.use_eos_token:
                    hidden, observed = hidden[:, :-1], observed[:, :-1]
                weights = observed[..., None].to(hidden.dtype)
                pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
                vectors.append(pooled.cpu().numpy())
        return np.concatenate(vectors).astype(float)

    def finetune(self, panel: TimeSeriesPanel, spec: TrainingSpec) -> dict[str, Any]:
        import torch

        module = self._model.model
        horizon = int(self._model.tokenizer.config.prediction_length)
        if spec.prediction_length != horizon:
            warn_once(
                f"Chronos v1 tokenises exactly {horizon} future steps, so fine-tuning uses "
                f"{horizon}-step windows rather than the requested {spec.prediction_length}"
                f"{' (shorter windows are padded and the padding is not scored)' if spec.prediction_length < horizon else ''}"
                f". Longer forecasts still work at predict time, where upstream rolls the "
                f"model forward.",
                UserWarning,
                key=f"chronos-train-horizon:{spec.prediction_length}",
            )
            spec = replace(spec, prediction_length=horizon)
        # AdamW on bfloat16 parameters is unsound, so train in float32 and cast back.
        module.float()
        try:
            report = super().finetune(panel, spec)
        finally:
            for p in module.parameters():
                p.requires_grad = False
            module.to(dtype=getattr(torch, self.dtype))
            module.eval()
        return {"objective": "token cross-entropy over the quantised future", **report}

    def _network(self) -> Any:
        return self._model.model

    def _training_context(self, spec: TrainingSpec) -> int:
        limit = int(self._model.tokenizer.config.context_length)
        return min(spec.context_length or limit, limit)

    def _training_loss(self, torch: Any, context: np.ndarray, label: np.ndarray) -> Any:
        """Chronos v1's pretraining objective: cross-entropy over the future tokens.

        The label is tokenised with the context's scale, so no ``context_scale``
        division applies on top. Unobserved future steps are masked with ``-100``.
        """
        tokenizer = self._model.tokenizer
        horizon = int(tokenizer.config.prediction_length)
        # label_input_transform asserts an exactly prediction_length-wide label.
        padded = np.full((len(label), horizon), np.nan, dtype=np.float32)
        width = min(label.shape[1], horizon)
        padded[:, :width] = label[:, :width]

        token_ids, attention_mask, scale = tokenizer.context_input_transform(
            torch.as_tensor(context, dtype=torch.float32)
        )
        label_ids, label_mask = tokenizer.label_input_transform(
            torch.as_tensor(padded, dtype=torch.float32), scale
        )
        labels = label_ids.clone()
        labels[~label_mask] = -100
        return self._model.model.model(
            input_ids=token_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
            labels=labels.to(self.device),
        ).loss

