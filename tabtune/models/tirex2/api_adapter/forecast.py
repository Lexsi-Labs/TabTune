# Copyright (c) NXAI GmbH.
# Licensed under the Apache License, Version 2.0; see LICENSE for details.

"""High-level forecasting API wrapping a :class:`TiRex2` backbone."""

import logging
from typing import Literal, get_args

import torch

from ..model.types import TimeseriesType

logger = logging.getLogger(__file__)

ForecastOutputType = Literal["torch", "numpy"]


def _is_oom_error(exc: BaseException) -> bool:
    """Return whether ``exc`` is an out-of-memory error, on CUDA or MPS.

    CUDA raises :class:`torch.cuda.OutOfMemoryError`; MPS currently surfaces OOM
    as a plain :class:`RuntimeError` whose message mentions running out of memory.
    """
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    if isinstance(exc, RuntimeError):
        return "out of memory" in str(exc).lower()
    return False


def _empty_device_cache(device: str) -> None:
    """Release the caching allocator for ``device`` (no-op on CPU)."""
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


def _format_output(forecasts, output_type):
    """Render a batch of per-series ``[V_t, Q, H]`` forecasts in the requested output format."""
    if output_type == "torch":
        return [f.cpu() for f in forecasts]
    elif output_type == "numpy":
        return [f.cpu().numpy() for f in forecasts]
    else:
        raise ValueError(f"Invalid output type: {output_type}")


def _predict_adaptive(
    model,
    timeseries,
    prediction_length,
    *,
    output_type,
    batch_size,
    **predict_kwargs,
):
    """Yield formatted forecasts batch by batch, halving the batch size on device OOM.

    Walks contiguous ``[start, end)`` windows of at most ``batch_size`` series,
    forecasting and formatting each. When a window runs out of memory (CUDA or MPS),
    the device cache is cleared, the batch size is halved (floor of 1), and the *same*
    window is retried at the smaller size. The reduced size persists for the rest of
    this call, so a single oversized window pins it down only here - a fresh call
    starts again from ``batch_size``. An OOM at size 1 is re-raised: a lone series
    that does not fit cannot be split further.

    Formatting - and the ``.cpu()`` move it performs - happens per window, so GPU
    memory backing completed batches is released as we go rather than accumulating
    across the whole dataset.
    """
    num_items = len(timeseries)
    device = str(getattr(model, "device", "cpu"))
    start = 0
    current = batch_size
    while start < num_items:
        end = min(start + current, num_items)
        try:
            forecasts = model.predict(timeseries[start:end], prediction_length, **predict_kwargs)
            formatted = _format_output(forecasts, output_type)
        except RuntimeError as exc:  # torch.cuda.OutOfMemoryError is a RuntimeError subclass
            if not _is_oom_error(exc):
                raise
            _empty_device_cache(device)
            if current == 1:
                logger.error("Device OOM at batch size 1 (series index %d); cannot shrink further.", start)
                raise
            current = max(1, current // 2)
            logger.warning("Device OOM at series index %d; halving batch size to %d and retrying.", start, current)
            continue
        yield formatted
        start = end


def _gen_forecast(
    model,
    timeseries,
    prediction_length,
    *,
    output_type,
    batch_size,
    yield_per_batch,
    **predict_kwargs,
):
    """Batch the timeseries, run :meth:`TiRex2.predict`, and accumulate or stream the formatted output.

    The batch size is reduced automatically on CUDA out-of-memory errors and
    reset for each call; see :func:`_predict_adaptive`.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}.")
    if output_type not in get_args(ForecastOutputType):
        raise ValueError(f"Invalid output type: {output_type!r}; expected one of {list(get_args(ForecastOutputType))}.")

    batch_outputs = _predict_adaptive(
        model,
        timeseries,
        prediction_length,
        output_type=output_type,
        batch_size=batch_size,
        **predict_kwargs,
    )

    if yield_per_batch:
        return batch_outputs

    all_forecasts = []
    for formatted in batch_outputs:
        all_forecasts.extend(formatted)
    return all_forecasts


class ForecastModel:
    """High-level, batched forecasting interface around a ``TiRex2`` backbone.

    The wrapper takes ownership of the model only as a delegate: it batches the
    ``TimeseriesType`` it is given, feeds them to ``TiRex2.predict``, and formats the
    per-series quantile forecasts into the requested output type. Attribute access falls
    through to the wrapped model, so the backbone's own methods (e.g. ``predict``) remain
    reachable on the wrapper.

    Parameters
    ----------
    model : TiRex2
        An instantiated, ready-for-inference backbone exposing
        ``predict(timeseries: list[TimeseriesType], prediction_length: int) -> list[Tensor]``
        and a ``quantiles`` buffer holding the quantile levels it forecasts.
    """

    def __init__(self, model):
        self.model = model

    def _quantile_levels(self) -> list[float]:
        """Return the model's forecast quantile levels as clean Python floats (float32 noise rounded off)."""
        return [round(float(q), 6) for q in self.model.quantiles]

    def __getattr__(self, name):
        """Delegate unknown attribute lookups to the wrapped model."""
        try:
            model = object.__getattribute__(self, "model")
        except AttributeError:
            raise AttributeError(name)
        return getattr(model, name)

    def forecast(
        self,
        timeseries: list[TimeseriesType],
        prediction_length: int,
        *,
        output_type: ForecastOutputType = "torch",
        batch_size: int = 512,
        yield_per_batch: bool = False,
        **predict_kwargs,
    ):
        """Forecast a list of ``TimeseriesType`` objects, each with a target and optional covariates.

        Returns one ``[V_t, Q, H]`` forecast per series (``Q`` quantile levels, see
        ``_quantile_levels``), as CPU torch tensors or numpy arrays.

        Extra ``predict_kwargs`` are forwarded verbatim to ``TiRex2.predict``.
        In particular ``tta_sign_flip`` controls sign-flip test-time augmentation
        (roughly doubles inference cost), and ``tta_diff`` controls postprocessor
        differencing; when omitted, the checkpoint's configured defaults
        (``model-config.yaml``) are used. Pass ``True``/``False`` to override.

        Examples
        --------
        >>> import torch
        >>> from tabtune.models.tirex2 import TimeseriesType, load_model
        >>> model = load_model("NX-AI/TiRex-2", device="cpu")
        >>> ts = TimeseriesType(target=torch.randn(1, 128), past_covariates=None, future_covariates=None)
        >>> forecasts = model.forecast([ts], prediction_length=32, output_type="numpy")
        >>> forecasts[0].shape
        (1, 9, 32)
        """
        return _gen_forecast(
            self.model,
            list(timeseries),
            prediction_length,
            output_type=output_type,
            batch_size=batch_size,
            yield_per_batch=yield_per_batch,
            **predict_kwargs,
        )

    def embed(
        self,
        timeseries: list[TimeseriesType],
        prediction_length: int | None = None,
        *,
        batch_size: int = 512,
        pad_context: bool = True,
    ) -> list[torch.Tensor]:
        """Return one ``[V_t, embedding_dim]`` CPU tensor per series; see :meth:`TiRex2.embed`."""
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}.")
        timeseries = list(timeseries)
        embeddings = []
        for start in range(0, len(timeseries), batch_size):
            batch = timeseries[start : start + batch_size]
            embeddings.extend(e.cpu() for e in self.model.embed(batch, prediction_length, pad_context=pad_context))
        return embeddings
