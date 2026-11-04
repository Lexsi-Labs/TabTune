# Portions derived from TabPFN Extensions:
# Copyright 2025 Prior Labs GmbH
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Any, cast

import torch
from torch import Tensor

from tabtune.models.kumo._sdm.cache import Cache


class ECOC(torch.nn.Module):
    """Extend a classifier with `error-correcting output codes <https://arxiv.org/abs/cs/9501101>`_.

    For more than ``max_classes`` classes, this uses the approach from
    `TabPFN Extensions <https://github.com/PriorLabs/tabpfn-extensions>`__:
    batched tasks each separate ``max_classes - 1`` classes and merge the rest.
    Scores average log probabilities over tasks where each class is separate.
    Apply softmax to normalize them.

    Args:
        max_classes: Number of classes supported by the model.
    """

    def __init__(self, max_classes: int) -> None:
        super().__init__()
        self.max_classes = max_classes

    def forward(
        self,
        model: torch.nn.Module,
        x: Tensor,
        y: Tensor,
        *,
        num_classes: int,
        num_members: int = 1,
        cache: Cache | None = None,
        generator: torch.Generator | None = None,
        **kwargs: Any,
    ) -> Tensor:
        """Predict scores over the original classes.

        Args:
            model: Classifier accepting ``model(x, y, **kwargs)`` with leading
                batch dimensions and returning ``[..., R_query, max_classes]``
                logits. Must accept ``cache`` when caching is used.
            x: Context then query features of shape ``[..., R, C]``, for
                ``R`` rows and ``C`` columns.
            y: Integer context labels in ``[0, num_classes)``, of shape
                ``[..., R_context]``.
            num_classes: Total number of target classes ``K``, including those
                absent from the context.
            num_members: Number of ensemble members stacked along the first
                dimension of ``x`` and ``y``. Each member draws its own
                codebook, in member order, as separate calls would.
            cache: Cache for model state and the codebook. On replay, pass
                query-only ``x``, an empty context axis in ``y``, and the same
                ``num_classes``.
            generator: Codebook sampling generator on ``x.device``. Ignored
                on cache replay.
            kwargs: Model arguments. Tensors must broadcast over the leading
                task dimension.

        Returns:
            Scores of shape ``[..., R_query, K]`` for the query rows: model
            logits within class capacity, otherwise averaged log probabilities.
        """
        if num_classes <= self.max_classes:
            if cache is not None:
                kwargs["cache"] = cache
            return model(x, y, **kwargs)[..., :num_classes]

        if cache is not None and cache.is_replaying:
            codebook = cast(Tensor, cache["ecoc_codebook"])
            if num_classes != codebook.size(-1):
                raise ValueError(
                    "'num_classes' must match the cached ECOC codebook "
                    f"(expected {codebook.size(-1)}, got {num_classes})"
                )
            kwargs["cache"] = cast(Cache, cache["ecoc_model"])
        elif num_members == 1:
            # [T, K]
            codebook = self._draw_codebook(num_classes, x.device, generator)
        else:
            # [E, T, K]
            codebook = torch.stack(
                [
                    self._draw_codebook(num_classes, x.device, generator)
                    for _ in range(num_members)
                ]
            )
        if cache is not None and not cache.is_replaying:
            cache["ecoc_codebook"] = codebook
            kwargs["cache"] = cache["ecoc_model"] = Cache()

        T = codebook.size(-2)
        if codebook.dim() == 2:
            y = codebook.index_select(dim=1, index=y.reshape(-1)).view(
                T, *y.shape
            )  # [T, ..., R_context]
        else:
            E = codebook.size(0)
            y = (
                codebook.gather(
                    dim=-1,
                    index=y.long().reshape(E, 1, -1).expand(E, T, -1),
                )  # [E, T, N]
                .view(E, T, *y.shape[1:])
                .movedim(1, 0)
            )  # [T, E, ..., R_context]
        logits = model(x=x.expand(T, *x.shape), y=y, **kwargs)
        if codebook.dim() == 2:
            index = codebook.view(T, *(1,) * (logits.dim() - 2), num_classes)
        else:
            index = codebook.movedim(1, 0).reshape(
                T, E, *(1,) * (logits.dim() - 3), num_classes
            )
        scores = logits.log_softmax(dim=-1).gather(
            dim=-1,
            index=index.expand(*logits.shape[:-1], num_classes),
        )  # [T, ..., R_query, K]
        active = index != self.max_classes - 1
        return scores.masked_fill(~active, 0).sum(dim=0) / active.sum(dim=0)

    def _draw_codebook(
        self,
        num_classes: int,
        device: torch.device,
        generator: torch.Generator | None,
    ) -> Tensor:
        rest_idx = self.max_classes - 1
        num_codes = max(
            # Give every class its own output in at least one task.
            math.ceil(num_classes / rest_idx),
            4 * math.ceil(math.log(num_classes, self.max_classes)),
        )
        # Bound the quadratic distance search for large targets.
        num_draws = 50 if num_classes <= 200 else 1
        codebook = torch.full(
            size=(num_draws, num_codes, num_classes),
            fill_value=rest_idx,
            dtype=torch.long,
            device=device,
        )
        coverage = torch.zeros(num_draws, num_classes, device=device)
        for code in codebook.unbind(dim=1):
            priority = coverage + 0.1 * torch.rand(
                num_draws, num_classes, device=device, generator=generator
            )
            chosen = priority.argsort(dim=-1)[:, :rest_idx]
            symbols = torch.rand(
                num_draws, rest_idx, device=device, generator=generator
            ).argsort(dim=-1)
            code.scatter_(dim=-1, index=chosen, src=symbols)
            coverage.scatter_add_(
                dim=-1,
                index=chosen,
                src=torch.ones_like(chosen, dtype=coverage.dtype),
            )

        if num_draws == 1:
            return codebook[0]

        distance = torch.zeros(
            num_draws,
            num_classes,
            num_classes,
            dtype=torch.long,
            device=device,
        )
        for code in codebook.unbind(dim=1):
            distance += code.unsqueeze(-1) != code.unsqueeze(-2)
        rows, cols = torch.triu_indices(
            num_classes, num_classes, offset=1, device=device
        )
        pairwise = distance[:, rows, cols]
        # Prefer minimum distance, breaking ties by total pairwise distance.
        score = pairwise.amin(dim=-1) * (
            pairwise.size(-1) * num_codes + 1
        ) + pairwise.sum(dim=-1)
        return codebook.index_select(0, score.argmax().view(1)).squeeze(0)
