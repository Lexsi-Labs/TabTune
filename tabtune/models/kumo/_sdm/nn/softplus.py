# Portions derived from TabFM:
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F
from torch import Tensor


class SoftplusScale(torch.nn.Module):
    r"""Apply a learned positive scale to the final input dimension.

    Each channel is multiplied by a positive factor parametrized as
    ``multiplier * softplus(weight)``.

    Args:
        channels: The number of input and output channels.
        multiplier: The fixed multiplier applied to every learned factor.
        weight_init: The initial values of ``weight``.
        device: The device.
        dtype: The parameter dtype.
    """

    def __init__(
        self,
        channels: int,
        multiplier: float = 1.0,
        weight_init: float = 0.0,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.multiplier = multiplier
        self.weight = torch.nn.Parameter(
            data=torch.full(
                size=(channels,),
                fill_value=weight_init,
                device=device,
                dtype=dtype,
            )
        )

    def forward(self, tensor: Tensor) -> Tensor:
        r"""The forward pass.

        Args:
            tensor: The input tensor.
        """
        scale = self.multiplier * F.softplus(self.weight.float())
        return tensor * scale.to(tensor.dtype)
