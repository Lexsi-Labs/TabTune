# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/)
# Copyright 2025 Datadog, Inc.
#
# TabTune vendoring note: this file is upstream's, with the one gluonts import rewritten to
# import from the sibling ``_gluonts`` module instead of gluonts, so that gluonts
# is not a TabTune dependency (it pins numpy<2.2, which would downgrade numpy for
# the whole library). Nothing else is changed. See ``_gluonts.py`` for the list of
# symbols and why it exists.

from dataclasses import dataclass
from typing import cast

import numpy as np
import torch
from einops import rearrange, repeat
from .._gluonts import AffineTransformed
from jaxtyping import Bool, Float, Int
from torch.distributions import Distribution

from ..data.util.dataset import (
    MaskedTimeseries,
    pad_array,
    pad_id_mask,
    replace_extreme_values,
)
from ..model.backbone import TotoBackbone


@dataclass(frozen=True)
class Forecast:
    mean: Float[torch.Tensor, "batch variate future_time_steps"]
    samples: Float[torch.Tensor, "batch variate future_time_steps samples"] | None = None

    def quantile(self, q: float | torch.Tensor) -> Float[torch.Tensor, "batch variate future_time_steps"]:
        """
        Compute the quantile of the forecast samples.
        """
        assert self.samples is not None, "samples must be provided to compute quantiles"
        assert isinstance(q, float) or isinstance(q, torch.Tensor), "q must be a float or a tensor"
        if isinstance(q, float):
            q = torch.tensor(q, device=self.samples.device, dtype=self.samples.dtype)
        return self.samples.quantile(q, dim=-1)

    @property
    def median(self) -> Float[torch.Tensor, "batch variate future_time_steps"]:
        """
        The median of the forecast samples.
        """
        return self.quantile(0.5)

    @property
    def std(self) -> Float[torch.Tensor, "batch variate future_time_steps"]:
        """
        Compute the standard deviation of the forecast samples.
        """
        assert self.samples is not None, "samples must be provided to compute standard deviation"
        return self.samples.std(dim=-1)


class TotoForecaster:
    """
    A forecaster class for the Toto model that handles autoregressive decoding for time series forecasting.

    This class wraps a TotoBackbone model and provides methods to generate forecasts for time series data.
    The forecasting process uses an autoregressive decoding algorithm:

    1. The model first processes the entire input context (historical data)
    2. For each future time step:
       - The model generates a distribution over possible values
       - Either the mean or random samples are drawn from this distribution
       - The generated value(s) are appended to the input sequence
       - The process repeats with this extended sequence

    When generating multiple samples (num_samples > 1), the model creates separate trajectories for each sample:
    - Each trajectory starts with the same historical context
    - As sampling progresses, each trajectory evolves independently
    - This results in num_samples different possible future paths
    - Samples can be processed in batches (samples_per_batch) to manage memory usage

    The forecaster efficiently reuses computation from the context processing phase using a key-value cache,
    which stores intermediate transformer attention states to avoid redundant computation.

    The forecaster handles data preprocessing, including padding to match the model's patch size,
    and postprocessing to format the outputs as a Forecast object containing means and optional samples.

    Exogenous Variables:
        When using exogenous variables, they MUST be placed at the END of the variates/channels dimension
        in the input tensor. For example, if you have 3 target variates and 2 exogenous variables,
        the input shape should be (batch, 5, time_steps) where:
        - indices 0, 1, 2 correspond to target variates
        - indices 3, 4 correspond to exogenous variables

        This convention allows the forecaster to correctly inject known future values of exogenous
        variables during autoregressive decoding by replacing the last `num_exogenous_variables`
        channels with the provided `future_exogenous_variables`.
    """

    model: TotoBackbone

    def __init__(
        self,
        model: TotoBackbone,
    ):
        self.model = model
        # set the model to evaluation mode
        self.model.eval()

    def forecast(
        self,
        inputs: MaskedTimeseries,
        prediction_length: int,
        num_samples: int | None = None,
        samples_per_batch: int = 10,
        use_kv_cache: bool = True,
        future_exogenous_variables: Float[torch.Tensor, "batch exogenous_variables future_time_steps"] | None = None,
    ) -> Forecast:
        """
        Generate a forecast for a batch of time series. This method works autoregressively,
        i.e. it feeds the model's predictions back into itself. The decoding process is as follows:

        1. The model first processes the entire input context (historical data)
        2. For each future time step:
            - The model generates a distribution over possible values
            - Either the mean or random samples are drawn from this distribution
            - If known future exogenous variables are provided, they are injected into the samples to replace the predicted values
            - The generated value(s) are appended to the input sequence
            - The process repeats with this extended sequence

        There are two modes of operation:
        1. num_samples is None: generate a single mean prediction
        2. num_samples is not None: generate num_samples random samples

        When num_samples is not None, the model creates num_samples separate trajectories for each sample:
        - Each trajectory starts with the same historical context
        - As sampling progresses, each trajectory evolves independently
        - This results in num_samples different possible future paths
        - Samples can be processed in batches (samples_per_batch) to manage memory usage

        When using samples_per_batch, this batch size compounds with the optional batch dimension of the input.
        For example, if you have a batch of 10 time series, and you set samples_per_batch to 10,
        the effective batch size is 100. For the best performance, set samples_per_batch
        as high as possible, subject to memory constraints.

        Args:
            inputs: A MaskedTimeseries object containing the input time series.
            prediction_length: The number of future time steps to predict.
            num_samples:
                The number of samples to generate.
                If None, a single mean prediction is generated. However,
                the mean point forecast tends to be less accurate than the
                median or mean of the samples (provided enough samples are generated).
                It's recommended to use at least 128 samples for reliable forecasts.
            samples_per_batch:
                The number of samples to generate per batch.
                In most cases, this should be as high as possible, subject to memory constraints.
                When the inputs have a batch dimension, the effective batch size is samples_per_batch * batch_size.
            use_kv_cache:
                Whether to use a key-value cache for the model. In most cases, this should be True,
                as it significantly speeds up inference.
            future_exogenous_variables:
                If known future exogenous variables are provided, they are injected into the samples
                to replace the predicted values after each prediction step.

                IMPORTANT: Exogenous variables MUST be placed at the END of the variates/channels
                dimension in the input tensor. The forecaster assumes that the last `num_exogenous_variables`
                channels in `inputs.series` correspond to exogenous variables. During autoregressive
                decoding, predictions for these channels are replaced with the known future values
                from `future_exogenous_variables`.
        """
        if len(inputs.series.shape) == 2:
            # unbatched input, variates x time_steps
            batch = cast(MaskedTimeseries, torch.utils.data.default_collate([inputs]))
        else:
            # input is already batched
            batch = inputs

        if future_exogenous_variables is not None and len(future_exogenous_variables.shape) == 2:
            future_exogenous_variables = future_exogenous_variables.unsqueeze(0)

        # pad the input to the nearest multiple of the patch size
        series = pad_array(batch.series, self.model.patch_embed.stride)
        padding_mask = pad_array(batch.padding_mask, self.model.patch_embed.stride)
        id_mask = batch.id_mask
        if id_mask is not None:
            id_mask = pad_id_mask(batch.id_mask, self.model.patch_embed.stride)
        timestamp_seconds = pad_array(batch.timestamp_seconds, self.model.patch_embed.stride)
        time_interval_seconds: Int[torch.Tensor, "batch variate series_len"] = torch.as_tensor(
            batch.time_interval_seconds, device=series.device, dtype=torch.int
        )

        if num_samples is not None:
            samples = self.generate_samples(
                inputs=series,
                prediction_length=prediction_length,
                num_samples=num_samples,
                timestamp_seconds=timestamp_seconds,
                time_interval_seconds=time_interval_seconds,
                input_padding_mask=padding_mask,
                id_mask=id_mask,
                sampling_batch_size=samples_per_batch,
                use_kv_cache=use_kv_cache,
                future_exogenous_variables=future_exogenous_variables,
                num_exogenous_variables=batch.num_exogenous_variables,
            )
            mean = samples.mean(dim=-1)
        else:
            mean = self.generate_mean(
                inputs=series,
                prediction_length=prediction_length,
                timestamp_seconds=timestamp_seconds,
                time_interval_seconds=time_interval_seconds,
                input_padding_mask=padding_mask,
                id_mask=id_mask,
                use_kv_cache=use_kv_cache,
                future_exogenous_variables=future_exogenous_variables,
                num_exogenous_variables=batch.num_exogenous_variables,
            )
            samples = None

        return Forecast(mean=mean, samples=samples)

    def assert_ev_compatibility(
        self,
        inputs: Float[torch.Tensor, "batch total_variate patch_time_steps"],
        future_exogenous_variables: Float[torch.Tensor, "batch exogenous_variables future_time_steps"],
        prediction_length: int,
        num_exogenous_variables: int,
    ) -> None:
        """
        Assert the compatibility of the future exogenous variables with the input.
        """
        assert (
            future_exogenous_variables.shape[-1] == prediction_length
        ), "The future exogenous variables must have the same length as the prediction length"
        assert (
            future_exogenous_variables.shape[0] == inputs.shape[0]
        ), "The future exogenous variables must have the same batch size as the input"
        assert (
            num_exogenous_variables == future_exogenous_variables.shape[-2]
        ), "The number of exogenous variables must match the number of exogenous variables in the future_exogenous_variables"

    def round_ft_ev(
        self,
        future_exogenous_variables: Float[torch.Tensor, "batch exogenous_variables future_time_steps"],
        T_rounded: int,
    ) -> Float[torch.Tensor, "batch exogenous_variables rounded_steps"]:
        # add padding to the future exogenous variables to the nearest multiple of the patch size
        B, V_ev, T_future = future_exogenous_variables.shape
        dtype = future_exogenous_variables.dtype
        device = future_exogenous_variables.device
        padding = torch.zeros(B, V_ev, T_rounded - T_future, device=device, dtype=dtype)
        padded_future_exogenous_variables = torch.cat([future_exogenous_variables, padding], dim=-1)
        return padded_future_exogenous_variables

    @torch.no_grad()
    def generate_mean(
        self,
        inputs: Float[torch.Tensor, "batch variate time_steps"],
        prediction_length: int,
        timestamp_seconds: Int[torch.Tensor, "batch variate time_steps"],
        time_interval_seconds: Int[torch.Tensor, "batch variate"],
        input_padding_mask: Bool[torch.Tensor, "batch variate time_steps"] | None = None,
        id_mask: Float[torch.Tensor, "batch #variate time_steps"] | None = None,
        use_kv_cache: bool = False,
        future_exogenous_variables: Float[torch.Tensor, "batch exogenous_variables future_time_steps"] | None = None,
        num_exogenous_variables: int = 0,
    ) -> Float[torch.Tensor, "batch variate time_steps"]:
        """
        Generate a point prediction by taking the mean of the output distribution at each step.
        This method works autoregressively, i.e. it feeds the model's predictions back into itself
        to generate the next prediction.

        If future exogenous variables are provided, they are injected into the samples to replace
        the predicted values after each prediction step.

        Note:
            Exogenous variables MUST be placed at the END of the variates/channels dimension.
            The last `num_exogenous_variables` channels are assumed to be exogenous variables
            and will be replaced with values from `future_exogenous_variables` during decoding.
        """
        if input_padding_mask is None:
            input_padding_mask = torch.ones_like(inputs, dtype=torch.bool, device=inputs.device)
        if id_mask is None:
            id_mask = torch.zeros_like(inputs, dtype=torch.int, device=inputs.device)

        # Assert the compatibility of the future exogenous variables with the input
        if future_exogenous_variables is not None:
            self.assert_ev_compatibility(inputs, future_exogenous_variables, prediction_length, num_exogenous_variables)

        ## round up the prediction length to the nearest multiple of the patch size
        patch_size = self.model.patch_embed.stride
        rounded_steps = int(np.ceil(prediction_length / patch_size) * patch_size)
        if rounded_steps > prediction_length and future_exogenous_variables is not None:
            future_exogenous_variables = self.round_ft_ev(future_exogenous_variables, rounded_steps)
        start_index = inputs.shape[-1]
        end_index = start_index + prediction_length

        # TODO: maybe pass in future masks, rather than making assumptions here?
        dummy_padding = torch.ones(
            (input_padding_mask.shape[0], input_padding_mask.shape[1], patch_size),
            device=inputs.device,
            dtype=torch.bool,
        )
        dummy_id_mask = repeat(
            id_mask[:, :, -1:],
            "batch variates 1 -> batch variates patch_size",
            patch_size=patch_size,
        )
        if use_kv_cache:
            kv_cache = self.model.allocate_kv_cache(
                batch_size=inputs.shape[0],
                num_variates=inputs.shape[1],
                max_time_steps=inputs.shape[2] + rounded_steps,
                device=inputs.device,
                dtype=inputs.dtype,
            )
        else:
            kv_cache = None

        scaling_prefix_length = inputs.shape[-1]

        for idx in range(rounded_steps // patch_size):
            base_distr, loc, scale = self.model(
                inputs=inputs,
                input_padding_mask=input_padding_mask,
                id_mask=id_mask,
                kv_cache=kv_cache,
                scaling_prefix_length=scaling_prefix_length,
                num_exogenous_variables=num_exogenous_variables,
            )
            distr = self.create_affine_transformed(base_distr, loc, scale)

            # We remove extreme values that can occur early in training
            # and cause validation metrics to be NaN
            samples = replace_extreme_values(distr.mean[:, :, -patch_size:])

            # If future exogenous variables are provided, inject them into the samples to replace
            # the predicted values. Note: exogenous variables are assumed to be the LAST channels
            # in the variates dimension, hence we use `[:, -num_exogenous_variables:]` indexing.
            if future_exogenous_variables is not None:
                start, stop = idx * patch_size, (idx + 1) * patch_size
                samples[:, -num_exogenous_variables:] = future_exogenous_variables[:, :, start:stop]

            inputs = torch.cat([inputs, samples], dim=-1)
            id_mask = torch.cat([id_mask, dummy_id_mask], dim=-1)
            input_padding_mask = torch.cat([input_padding_mask, dummy_padding], dim=-1)
            for _ in range(patch_size):
                next_timestamp: Int[torch.Tensor, "batch variate"] = timestamp_seconds[:, :, -1] + time_interval_seconds
                timestamp_seconds = torch.cat([timestamp_seconds, next_timestamp.unsqueeze(-1)], dim=-1)

        return inputs.detach()[:, :, start_index:end_index]

    @torch.no_grad()
    def generate_samples(
        self,
        inputs: Float[torch.Tensor, "batch variate time_steps"],
        prediction_length: int,
        num_samples: int,
        timestamp_seconds: Int[torch.Tensor, "batch variate time_steps"],
        time_interval_seconds: Int[torch.Tensor, "batch variate"],
        input_padding_mask: Bool[torch.Tensor, "batch variate time_steps"] | None = None,
        id_mask: Float[torch.Tensor, "batch #variate time_steps"] | None = None,
        sampling_batch_size: int = 10,
        use_kv_cache: bool = False,
        future_exogenous_variables: Float[torch.Tensor, "batch exogenous_variables future_time_steps"] | None = None,
        num_exogenous_variables: int = 0,
    ) -> Float[torch.Tensor, "batch variate time_steps samples"]:
        """
        Generate samples from the output distribution.
        This method works autorregressively, i.e. it feeds the model's predictions back into itself.
        It works by creating num_samples chains. Each chain is a separate sequence of predictions.
        At each time step, for each chain we take a single sample from the output distribution and append
        it to the end of the sequence.

        If future exogenous variables are provided, they are injected into the samples to replace
        the predicted values after each prediction step.

        Note:
            Exogenous variables MUST be placed at the END of the variates/channels dimension.
            The last `num_exogenous_variables` channels are assumed to be exogenous variables
            and will be replaced with values from `future_exogenous_variables` during decoding.
        """
        if input_padding_mask is None:
            input_padding_mask = torch.ones_like(inputs, dtype=torch.bool, device=inputs.device)
        if id_mask is None:
            id_mask = torch.zeros_like(inputs, dtype=torch.int, device=inputs.device)

        if future_exogenous_variables is not None:
            self.assert_ev_compatibility(inputs, future_exogenous_variables, prediction_length, num_exogenous_variables)

        assert num_samples % sampling_batch_size == 0, "num_samples must be divisible by sampling_batch_size"
        num_batches = num_samples // sampling_batch_size

        # round up the prediction length to the nearest multiple of the patch size
        patch_size = self.model.patch_embed.patch_size
        rounded_steps = int(np.ceil(prediction_length / patch_size) * patch_size)
        if rounded_steps > prediction_length and future_exogenous_variables is not None:
            future_exogenous_variables = self.round_ft_ev(future_exogenous_variables, rounded_steps)
        start_index = inputs.shape[-1]
        end_index = start_index + prediction_length

        dummy_padding = torch.ones(
            (
                input_padding_mask.shape[0] * sampling_batch_size,
                input_padding_mask.shape[1],
                patch_size,
            ),
            dtype=torch.bool,
            device=inputs.device,
        )

        dummy_id_mask = repeat(
            id_mask[:, :, -1:],
            "batch variates 1 -> (sampling_batch_size batch) variates patch_size",
            sampling_batch_size=sampling_batch_size,
            patch_size=patch_size,
        )
        inputs = repeat(
            inputs,
            "batch variates seq_len -> (sampling_batch_size batch) variates seq_len",
            sampling_batch_size=sampling_batch_size,
        )
        if future_exogenous_variables is not None:
            future_exogenous_variables = repeat(
                future_exogenous_variables,
                "batch exogenous_variables future_time_steps -> (sampling_batch_size batch) exogenous_variables future_time_steps",
                sampling_batch_size=sampling_batch_size,
            )
        input_padding_mask = repeat(
            input_padding_mask,
            "batch variates seq_len -> (sampling_batch_size batch) variates seq_len",
            sampling_batch_size=sampling_batch_size,
        )
        id_mask = repeat(
            id_mask,
            "batch variates seq_len -> (sampling_batch_size batch) variates seq_len",
            sampling_batch_size=sampling_batch_size,
        )
        timestamp_seconds = repeat(
            timestamp_seconds,
            "batch variates seq_len -> (sampling_batch_size batch) variates seq_len",
            sampling_batch_size=sampling_batch_size,
        )
        time_interval_seconds = repeat(
            time_interval_seconds,
            "batch variates -> (sampling_batch_size batch) variates",
            sampling_batch_size=sampling_batch_size,
        )

        all_samples = []
        if use_kv_cache:
            kv_cache = self.model.allocate_kv_cache(
                batch_size=inputs.shape[0],
                num_variates=inputs.shape[1],
                max_time_steps=inputs.shape[2] + rounded_steps,
                device=inputs.device,
                dtype=inputs.dtype,
            )
        else:
            kv_cache = None

        scaling_prefix_length = inputs.shape[-1]

        for _ in range(num_batches):
            batch_inputs = torch.clone(inputs)
            batch_input_padding_mask = torch.clone(input_padding_mask)
            batch_id_mask = torch.clone(id_mask)
            batch_timestamp_seconds = torch.clone(timestamp_seconds)

            for idx in range(rounded_steps // patch_size):
                base_distr, loc, scale = self.model(
                    inputs=batch_inputs,
                    input_padding_mask=batch_input_padding_mask,
                    id_mask=batch_id_mask,
                    kv_cache=kv_cache,
                    scaling_prefix_length=scaling_prefix_length,
                    num_exogenous_variables=num_exogenous_variables,
                )
                distr = self.create_affine_transformed(base_distr, loc, scale)

                sample = distr.sample()
                assert sample is not None

                # We remove extreme values that can occur early in training
                # and cause validation metrics to be NaN
                samples = replace_extreme_values(sample[:, :, -patch_size:])

                # If future exogenous variables are provided, inject them into the samples to replace
                # the predicted values. Note: exogenous variables are assumed to be the LAST channels
                # in the variates dimension, hence we use `[:, -num_exogenous_variables:]` indexing.
                if future_exogenous_variables is not None:
                    start, stop = idx * patch_size, (idx + 1) * patch_size
                    samples[:, -num_exogenous_variables:] = future_exogenous_variables[:, :, start:stop]
                batch_inputs = torch.cat([batch_inputs, samples], dim=-1)
                batch_id_mask = torch.cat([batch_id_mask, dummy_id_mask], dim=-1)
                batch_input_padding_mask = torch.cat([batch_input_padding_mask, dummy_padding], dim=-1)
                for _ in range(patch_size):
                    next_timestamp = batch_timestamp_seconds[:, :, -1] + time_interval_seconds
                    batch_timestamp_seconds = torch.cat([batch_timestamp_seconds, next_timestamp.unsqueeze(-1)], dim=-1)
            all_samples.append(batch_inputs)
            if kv_cache is not None:
                kv_cache.reset()

        outputs = torch.cat(all_samples, dim=0)
        unfolded_outputs = rearrange(
            outputs,
            "(samples batch) variates seq_len -> batch variates seq_len samples",
            samples=num_samples,
        ).detach()

        trimmed_predictions = unfolded_outputs[:, :, start_index:end_index, :]
        return trimmed_predictions

    @staticmethod
    def create_affine_transformed(base_distr: Distribution, loc: torch.Tensor, scale: torch.Tensor) -> Distribution:
        """
        Creates an AffineTransformed distribution with correctly matched shapes.

        Handles three cases:
        1. When loc/scale are per-timestep (from CausalStdMeanScaler)
        2. When base_distr only contains the distribution for the latest patch
           while loc/scale contain values for the entire sequence
        3. When loc/scale have a single time step (from StdMeanScaler/StdMinScaler)
           and need to be broadcast to match a multi-step base distribution

        Args:
            base_distr: The base distribution to transform
            loc: Location parameter
            scale: Scale parameter

        Returns:
            An AffineTransformed distribution with properly handled shapes
        """
        # Get the shape of the base distribution
        # We'll use this to match the time dimension of loc/scale
        base_shape = base_distr.mean.shape

        base_time_dim = base_shape[-1]  # Time dimension of base distribution
        loc_time_dim = loc.shape[-1]  # Time dimension of loc

        if loc_time_dim == 1:
            # Case 1: If loc/scale have time dimension 1 (standard scalers), PyTorch broadcasting will handle it
            return AffineTransformed(base_distr, loc=loc, scale=scale)

        # Case 2: If loc/scale have time dimension > 1 (causal scaler with history)
        # We need to extract only the suffix that matches the base distribution
        return AffineTransformed(base_distr, loc=loc[:, :, -base_time_dim:], scale=scale[:, :, -base_time_dim:])
