# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Set-transformer modules for structured tensor models."""

from __future__ import annotations

from typing import Any, Literal, overload

import torch
from torch import Tensor
from torch.nn import Parameter

from tabtune.models.kumo._sdm.cache import KVCacheEntry
from tabtune.models.kumo._sdm.nn import TransformerBlock


class InducedTransformerBlock(torch.nn.Module):
    r"""Transformer block using learned inducing points.

    Introduced in the `"Set Transformer: A Framework for Attention-based
    Permutation-Invariant Neural Networks"
    <https://arxiv.org/abs/1810.00825>`_ paper, the block routes attention
    through a small set of :math:`M` learned inducing points, reducing the cost
    of attending a query of size :math:`Q` to a key/value context of size
    :math:`K` from :math:`O(Q \cdot K)` to :math:`O((Q + K) \cdot M)`:

    .. math::

        H = \mathrm{InducingBlock}_1(I, \mathrm{key\_value}), \quad
        \mathrm{out} = \mathrm{OutputBlock}_2(\mathrm{query}, H),

    where :math:`I` are the learned inducing points and :math:`H` are the
    inducing points after attending to the key/value elements. Passing
    ``key_value=None`` recovers the induced self-attention block (ISAB).

    Args:
        channels: The number of input and output channels.
        num_inducing_points: The number of learned inducing points :math:`M`.
        inducing_block: The :class:`TransformerBlock` that updates the learned
            inducing points from key/value context.
        output_block: The :class:`TransformerBlock` that updates the input
            queries from the induced context.
        device: The device.
        dtype: The dtype.
    """

    def __init__(
        self,
        channels: int,
        num_inducing_points: int,
        inducing_block: TransformerBlock,
        output_block: TransformerBlock,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}

        self.inducing_block = inducing_block
        self.output_block = output_block

        self.inducing_points = Parameter(
            torch.empty(num_inducing_points, channels, **factory_kwargs)
        )
        torch.nn.init.trunc_normal_(self.inducing_points, std=0.02)

    @overload
    def forward(
        self,
        query: Tensor,
        key_value: Tensor | KVCacheEntry | None = None,
        seqused_key_value: Tensor | None = None,
        attn_mask: Tensor | None = None,
        *,
        return_key_value: Literal[False] = False,
        batch_size_limit: int | Literal["auto"] | None = None,
        out: Tensor | None = None,
    ) -> Tensor: ...

    @overload
    def forward(
        self,
        query: Tensor,
        key_value: Tensor | KVCacheEntry | None = None,
        seqused_key_value: Tensor | None = None,
        attn_mask: Tensor | None = None,
        *,
        return_key_value: Literal[True],
        batch_size_limit: int | Literal["auto"] | None = None,
        out: Tensor | None = None,
    ) -> tuple[Tensor, KVCacheEntry]: ...

    @overload
    def forward(
        self,
        query: Tensor,
        key_value: Tensor | KVCacheEntry | None = None,
        seqused_key_value: Tensor | None = None,
        attn_mask: Tensor | None = None,
        *,
        return_key_value: bool,
        batch_size_limit: int | Literal["auto"] | None = None,
        out: Tensor | None = None,
    ) -> Tensor | tuple[Tensor, KVCacheEntry]: ...

    def forward(
        self,
        query: Tensor,  # [..., Q, C]
        key_value: Tensor | KVCacheEntry | None = None,  # [..., KV, C]
        seqused_key_value: Tensor | None = None,  # [...]
        attn_mask: Tensor | None = None,  # [..., KV]
        *,
        return_key_value: bool = False,
        batch_size_limit: int | Literal["auto"] | None = None,
        out: Tensor | None = None,
    ) -> Tensor | tuple[Tensor, KVCacheEntry]:  # [..., Q, C]
        r"""The forward pass.

        Args:
            query: The query tensor with shape ``[..., Q, C]``.
                ``Q`` is the query sequence length, ``C`` is the number of
                channels.
            key_value: The key/value tensor with shape ``[..., KV, C]`` or
                precomputed key/value projections as a
                :class:`~sdm.cache.KVCacheEntry`.
                ``KV`` is the key/value sequence length.
                If omitted, ``query`` is used for induced self-attention.
            seqused_key_value: Valid key/value lengths with shape ``[...]`` and
                :external+torch:ref:`torch.int32 <dtype-doc>` dtype.
            attn_mask: Boolean attention mask with shape ``[..., KV]``.
                Entries set to ``True`` participate in attention.
            return_key_value: Whether to return the computed key and value
                projections for the final attention site alongside the output.
            batch_size_limit: Maximum number of batch elements processed at
                once.
            out: The output tensor.

        Returns:
            Tensor with shape ``[..., Q, C]`` when ``return_key_value`` is
            ``False``. Otherwise, a tuple of the output tensor and a
            :class:`~sdm.cache.KVCacheEntry`.
        """
        if not isinstance(key_value, KVCacheEntry):
            if key_value is None:
                key_value = query
            if attn_mask is not None:
                attn_mask = attn_mask.unsqueeze(-2)  # [..., 1, KV]
            key_value = self.inducing_block(
                query=self.inducing_points,  # [M, C]
                key_value=key_value,  # [..., KV, C]
                seqused_key_value=seqused_key_value,  # [...]
                attn_mask=attn_mask,  # [..., 1, KV]
                batch_size_limit=batch_size_limit,
            )  # [..., M, C]
        return self.output_block(
            query=query,  # [..., Q, C]
            key_value=key_value,  # [..., M, C]
            return_key_value=return_key_value,
            batch_size_limit=batch_size_limit,
            out=out,
        )  # [..., Q, C]
