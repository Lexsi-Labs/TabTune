# Copyright (c) NXAI GmbH.
# This software may be used and distributed according to the terms of the NXAI Community License Agreement.

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Union

import numpy as np
import torch

ContextType = Union[
    torch.Tensor,
    np.ndarray,
    list[torch.Tensor],
    list[np.ndarray],
]


def _ensure_1d_tensor(sample) -> torch.Tensor:
    if isinstance(sample, torch.Tensor):
        tensor = sample
    else:
        tensor = torch.as_tensor(sample)

    if tensor.ndim > 1:
        tensor = tensor.squeeze()

    assert tensor.ndim == 1, "Each sample must be one-dimensional"
    return tensor


def _pad_time_series_batch(
    batch_series: list[torch.Tensor],
    max_length: int,
) -> torch.Tensor:
    if not batch_series:
        return torch.empty((0, max_length))

    first = batch_series[0]
    dtype = first.dtype if first.is_floating_point() else torch.float32
    device = first.device

    padded = torch.full((len(batch_series), max_length), float("nan"), dtype=dtype, device=device)

    for idx, series in enumerate(batch_series):
        series = series.to(padded.dtype)
        series_len = series.shape[0]
        padded[idx, max_length - series_len :] = series

    return padded


def get_batches(context: ContextType, batch_size: int) -> Iterator[list[torch.Tensor]]:
    if isinstance(context, (torch.Tensor, np.ndarray)):
        context = torch.as_tensor(context)
        if context.ndim == 1:
            context = context.unsqueeze(0)
        assert context.ndim == 2
    elif not isinstance(context, (list, tuple)):
        raise ValueError(f"Context type {type(context)} not supported! Supported Types: {ContextType}")

    for start in range(0, len(context), batch_size):
        yield [_ensure_1d_tensor(sample) for sample in context[start : start + batch_size]]


def _call_fc_with_padding(fc_func, batch_series: list[torch.Tensor], **predict_kwargs):
    if not batch_series:
        raise ValueError("Received empty batch for forecasting")

    max_len = max(series.shape[0] for series in batch_series)
    padded_ts = _pad_time_series_batch(batch_series, max_len)

    return fc_func(padded_ts, **predict_kwargs)


class ForecastModel(ABC):
    @abstractmethod
    def _forecast_quantiles(self, batch, **predict_kwargs):
        pass

    def forecast(
        self,
        context: ContextType,
        prediction_length: int | None = None,
        batch_size: int = 512,
        full_rollout: bool = False,
        dynamic_padding: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        This method takes historical context data as input and outputs probabilistic forecasts.

        Args:
            context (ContextType): The historical "context" data of the time series:
                - `torch.Tensor`: 1D `[context_length]` or 2D `[batch_dim, context_length]` tensor
                - `np.ndarray`: 1D `[context_length]` or 2D `[batch_dim, context_length]` array
                - `List[torch.Tensor]`: List of 1D tensors (samples with different lengths get padded per batch)
                - `List[np.ndarray]`: List of 1D arrays (samples with different lengths get padded per batch)

            prediction_length (int, optional): Number of steps to forecast. Defaults to one output patch.

            batch_size (int, optional): The number of time series instances to process concurrently by the model.
                                        Defaults to 512. Must be $>= 1$.

            full_rollout (bool, optional): How the forecast horizon is generated:
                - `False`: One patch is predicted per forward pass.
                - `True`: The whole horizon is predicted in a single forward pass. This is faster for
                  horizons longer than one patch, but forecasting quality might degrade as the horizon grows.
                Defaults to `False`.

            dynamic_padding (bool, optional): How the context is padded before each forward pass:
                - `False`: The context is padded to the full training context length.
                - `True`: The context is padded to the smallest multiple of the patch size that fits it.
                  This is faster for contexts shorter than the training context length, but forecasting quality might degrade.
                Defaults to `False`.

        Returns:
            `Tuple[torch.Tensor, torch.Tensor]` (quantiles, mean) on CPU. The quantiles have the shape
            [batch_dim, forecast_len, quantile_count]; the mean is the median quantile.
        """
        assert batch_size >= 1, "Batch size must be >= 1"
        prediction_q = []
        prediction_m = []
        for batch_series in get_batches(context, batch_size):
            quantiles, mean = _call_fc_with_padding(
                self._forecast_quantiles,
                batch_series,
                prediction_length=prediction_length,
                full_rollout=full_rollout,
                dynamic_padding=dynamic_padding,
            )
            prediction_q.append(quantiles)
            prediction_m.append(mean)

        return torch.cat(prediction_q, dim=0).cpu(), torch.cat(prediction_m, dim=0).cpu()
