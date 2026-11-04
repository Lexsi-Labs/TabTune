# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D205
from __future__ import annotations

from collections.abc import Iterable
from typing import Any, ClassVar, Literal, cast

import torch
from torch import Tensor
from torch.nn import Identity, Linear, ModuleDict

from tabtune.models.kumo._sdm import Recipe, RelatedTables, Stype, TableTensor, Task, TaskLike
from tabtune.models.kumo._sdm.cache import Cache
from tabtune.models.kumo._sdm.models import ECOC, ICLModel
from tabtune.models.kumo._sdm.models._huggingface import download_checkpoint
from tabtune.models.kumo._sdm.models.kumo.tabular.icl import ICLBlock
from tabtune.models.kumo._sdm.models.kumo.tabular.recipe import default_recipe
from tabtune.models.kumo._sdm.models.kumo.tabular.row_embedding import RowEmbedding

MODEL_KWARGS: dict[str, dict[str, Any]] = {
    "small": {
        "cell_channels": 128,
        "num_embedding_layers": 4,
        "num_embedding_heads": 4,
        "num_inducing_points": 128,
        "group_size": 3,
        "num_frequencies": 32,
        "num_readout_tokens": 4,
        "icl_channels": 512,
        "num_icl_layers": 12,
        "num_icl_heads": 8,
        "num_icl_key_value_heads_for_query": None,
    },
    "medium": {
        "cell_channels": 256,
        "num_embedding_layers": 6,
        "num_embedding_heads": 4,
        "num_inducing_points": 256,
        "group_size": 3,
        "num_frequencies": 32,
        "num_readout_tokens": 4,
        "icl_channels": 512,
        "num_icl_layers": 24,
        "num_icl_heads": 8,
        "num_icl_key_value_heads_for_query": 2,
    },
    "large": {
        "cell_channels": 256,
        "num_embedding_layers": 6,
        "num_embedding_heads": 4,
        "num_inducing_points": 256,
        "group_size": 3,
        "num_frequencies": 32,
        "num_readout_tokens": 4,
        "icl_channels": 1024,
        "num_icl_layers": 24,
        "num_icl_heads": 16,
        "num_icl_key_value_heads_for_query": 2,
    },
}


class KumoTabular(ICLModel):
    r"""The tabular foundation model from `"NVIDIA Kumo Tabular Sets a New
    Accuracy-Efficiency Frontier for Tabular Prediction"
    <https://huggingface.co/blog/nvidia/kumo-tabular>`__.

    .. figure:: /images/kumo_tabular.svg
        :width: 100%

    :class:`KumoTabular` processes a table in two stages. First, an interleaved
    row/column encoder transforms raw table cells into fixed-size row
    representations. Numerical and categorical cells use separate learned
    Fourier frequencies and projections, together with a learned missingness
    projection. Each encoder stage consists of:

    * **Column-wise:** Each feature group is processed across rows using
      induced set attention. Both context and query cells attend only to
      context-row keys and values.
    * **Row-wise:** Each row's feature groups and learnable readout tokens
      attend to one another, combining feature interactions into a fixed-size
      row representation.

    Second, a dataset-wise in-context learning transformer processes the row
    representations and predicts each query target from the labeled context
    rows. The ``"large"`` model widens this transformer to 1024 channels and
    16 attention heads. Its four 256-channel readout tokens concatenate
    directly to that width without a projection layer.

    For regression tasks, :class:`KumoTabular` predicts 999 quantiles named
    ``"q001"`` through ``"q999"``.

    Args:
        task: The tasks to initialize. If ``None``, all tasks supported by this
            model are initialized.
        size: The model size, one of ``"small"``, ``"medium"``, or
            ``"large"``. Defaults to ``"large"``.
        pretrained: Whether to load pretrained checkpoints.
        device: The device for model parameters. If ``None``, uses PyTorch's
            default device.
    """

    supported_feature_stypes: ClassVar[frozenset[Stype]] = frozenset(
        {Stype.numerical}
    )
    supported_target_stypes: ClassVar[frozenset[Stype]] = frozenset(
        {Stype.numerical, Stype.categorical}
    )
    supports_multi_target: ClassVar[bool] = False
    supports_related_tables: ClassVar[bool] = False

    def __init__(
        self,
        task: TaskLike | Iterable[TaskLike] | None = None,
        size: Literal["small", "medium", "large"] = "large",
        pretrained: bool = True,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__(task=task)
        if Task.classification in self.tasks:
            self.ecoc = ECOC(max_classes=10)

        self.models: ModuleDict[TaskLike, torch.nn.Module] = ModuleDict()
        for task in self.tasks:
            self.models[task] = _KumoTabular(
                num_classes=10 if task == Task.classification else 0,
                num_quantiles=999 if task == Task.regression else 0,
                device="meta" if pretrained else device,
                **MODEL_KWARGS[size],
            )

        if pretrained:
            self.models[task] = self._load_from_pretrained(size, device=device)

        self.eval()

    @classmethod
    def default_recipe(cls) -> Recipe:
        r""":meta private:"""  # noqa: D415
        return default_recipe()

    def _load_from_pretrained(
        self,
        size: Literal["small", "medium", "large"],
        device: torch.device | str | None,
    ) -> _KumoTabular:
        device = torch.get_default_device() if device is None else device

        for task, model in self.models.items():
            if task == Task.classification:
                filename = f"{size}/classifier.pt"
            else:
                assert task == Task.regression
                filename = f"{size}/regressor.pt"

            path = download_checkpoint(
                repo_id="nvidia/Kumo-Tabular",
                filename=filename,
                revision="v1.0.0",
            )
            ckpt = torch.load(path, map_location=device, weights_only=True)
            model.load_state_dict(ckpt, assign=True)

        return model

    def _forward(
        self,
        x_context: TableTensor | None,  # [..., R_context, D]
        y_context: TableTensor | None,  # [..., R_context, 1]
        x_query: TableTensor | None,  # [..., R_query, D]
        related_context_tables: RelatedTables[TableTensor] | None,
        related_query_tables: RelatedTables[TableTensor] | None,
        cache: Cache | None,
        generator: torch.Generator | None,
        *,
        categorical_mask: Tensor,
        **kwargs: Any,
    ) -> TableTensor:  # [..., R_query, num_classes or 999]

        if x_query is None and x_context is not None:
            x = x_context.numerical
        elif x_context is None and x_query is not None:
            x = x_query.numerical
        else:
            assert x_context is not None
            assert x_query is not None
            x = torch.cat([x_context.numerical, x_query.numerical], dim=-2)

        classes: Tensor | None = None
        if y_context is not None and y_context.categorical.size(-1) > 0:
            y = y_context.categorical.code.squeeze(-1)
            classes = y_context.categorical.categories[0]
        elif y_context is not None and y_context.numerical.size(-1) > 0:
            y = y_context.numerical.squeeze(-1)
        else:
            assert cache is not None
            classes = cast(Tensor | None, cache["classes"])
            y = x.new_empty(
                (*x.size()[:-2], 0),
                dtype=torch.int64 if classes is not None else x.dtype,
            )

        categorical_mask = categorical_mask.expand(*x.size()[:-2], -1)

        if classes is None:
            out = self.models[Task.regression](
                x=x,
                y=y,
                categorical_mask=categorical_mask,
                cache=cache,
            )
            return TableTensor(
                columns={
                    Stype.numerical: [f"q{i:03d}" for i in range(1, 1000)]
                },
                numerical=out,
            )

        out = self.ecoc(
            model=self.models[Task.classification],
            x=x,
            y=y,
            num_classes=len(classes),
            num_members=x.size(0) if x.dim() > 2 else 1,
            cache=cache,
            generator=generator,
            categorical_mask=categorical_mask,
        )
        return TableTensor(
            columns={Stype.numerical: [str(i) for i in classes.tolist()]},
            numerical=out,
        )


class _KumoTabular(torch.nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_quantiles: int,
        cell_channels: int = 128,
        num_embedding_layers: int = 4,
        num_embedding_heads: int = 4,
        num_inducing_points: int = 128,
        group_size: int = 3,
        num_frequencies: int = 32,
        num_readout_tokens: int = 4,
        icl_channels: int = 512,
        num_icl_layers: int = 12,
        num_icl_heads: int = 8,
        num_icl_key_value_heads_for_query: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}

        self.row_embedding = RowEmbedding(
            num_classes=num_classes,
            channels=cell_channels,
            num_layers=num_embedding_layers,
            num_heads=num_embedding_heads,
            group_size=group_size,
            num_frequencies=num_frequencies,
            num_inducing_points=num_inducing_points,
            num_readout_tokens=num_readout_tokens,
            **factory_kwargs,
        )
        if cell_channels * num_readout_tokens != icl_channels:
            self.row_project = Linear(
                cell_channels * num_readout_tokens,
                icl_channels,
                **factory_kwargs,
            )
        else:
            self.row_project = Identity()
        self.icl_block = ICLBlock(
            num_classes=num_classes,
            out_channels=num_classes or num_quantiles,
            channels=icl_channels,
            num_layers=num_icl_layers,
            num_heads=num_icl_heads,
            num_key_value_heads_for_query=num_icl_key_value_heads_for_query,
            **factory_kwargs,
        )

    def forward(
        self,
        x: Tensor,  # [..., R, C]
        y: Tensor,  # [..., R_train]
        categorical_mask: Tensor,  # [..., C]
        *,
        cache: Cache | None = None,
    ) -> Tensor:  # [..., R_test, num_classes or num_quantiles]
        x = self.row_embedding(x, y, categorical_mask, cache=cache)
        x = self.row_project(x)
        return self.icl_block(x=x, y=y, cache=cache)
