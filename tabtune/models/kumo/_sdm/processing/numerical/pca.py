# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import cast

import torch

from tabtune.models.kumo._sdm import Stype, TableTensor
from tabtune.models.kumo._sdm.processing import Processor


class PCA(Processor):
    r"""Project numerical columns onto their principal components.

    Args:
        num_components: Number of principal components to keep.
    """

    handles_stypes = frozenset({Stype.numerical})
    requires_fit = True

    def __init__(self, num_components: int) -> None:
        super().__init__()
        self.num_components = num_components
        self.register_buffer("mean", torch.empty(0))
        self.register_buffer("components", torch.empty(0))

    def _fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:

        if table.numerical.size(-1) == 0:
            raise ValueError(
                f"{self.__class__.__name__!r} requires 'table' to have "
                f"numerical features"
            )

        self.mean = table.numerical.mean(dim=-2, keepdim=True)
        _, _, vh = torch.linalg.svd(
            table.numerical - self.mean,
            full_matrices=False,
        )
        num_components = min(
            self.num_components,
            table.numerical.size(-2),
            table.numerical.size(-1),
        )
        self.components = vh[..., :num_components, :].transpose(-2, -1)

    def _transform(self, table: TableTensor) -> TableTensor:
        x = (table.numerical - self.mean) @ self.components
        out = TableTensor(
            columns={Stype.numerical: [f"pca_{i}" for i in range(x.size(-1))]},
            numerical=x,
        )
        return cast(
            TableTensor,
            torch.cat([table.drop_stypes(Stype.numerical), out], dim=-1),
        )
