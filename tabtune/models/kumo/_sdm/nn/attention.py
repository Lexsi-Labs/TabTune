# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Attention modules for structured tensor models."""

import math
from typing import Any, Literal, overload

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear

from tabtune.models.kumo._sdm._memory import chunk_memory_limit
from tabtune.models.kumo._sdm.cache import KVCacheEntry
from tabtune.models.kumo._sdm.nn import QueryScaling


class SDPA(torch.nn.Module):
    r"""Scaled Dot-Product Attention (SDPA).

    This module wraps :func:`torch.nn.functional.scaled_dot_product_attention`
    and extends it by arbitrary batch dimensions, optional inference-time
    batch chunking, query-scaling, and padding support for key/value pairs.

    Args:
        num_query_heads: The number of query attention heads.
        num_key_value_heads: The number of key/value attention heads.
            Setting this below ``num_query_heads`` enables grouped-query
            attention (GQA); setting it to ``1`` enables multi-query attention
            (MQA). Must divide ``num_query_heads``. Defaults to
            ``num_query_heads`` (standard multi-head attention).
        query_scaling: Query scaling module to scale projected query heads
            before scaled dot-product attention, *e.g.*, :class:`QASSMax`.
        scale: Scaling factor passed to
            :func:`torch.nn.functional.scaled_dot_product_attention`.
            ``None`` uses the default value of ``1 / sqrt(channels)``.
    """

    def __init__(
        self,
        num_query_heads: int,
        num_key_value_heads: int | None = None,
        query_scaling: QueryScaling | None = None,
        scale: float | None = None,
    ) -> None:
        super().__init__()
        if num_key_value_heads is None:
            num_key_value_heads = num_query_heads
        if num_query_heads % num_key_value_heads != 0:
            raise ValueError(
                f"`num_query_heads` ({num_query_heads}) must be divisible by "
                f"`num_key_value_heads` ({num_key_value_heads})"
            )

        self.num_query_heads = num_query_heads
        self.num_key_value_heads = num_key_value_heads
        self.query_scaling = query_scaling
        self.scale = scale

    def forward(
        self,
        query: Tensor,  # [..., Q, Hq, C]
        key: Tensor,  # [..., KV, Hkv, C]
        value: Tensor,  # [..., KV, Hkv, C]
        seqused_key_value: Tensor | None = None,  # [...]
        attn_mask: Tensor | None = None,  # [..., Q, KV]
    ) -> Tensor:  # [..., Q, Hq, C]
        r"""The forward pass.

        Args:
            query: The query tensor with shape ``[..., Q, Hq, C]``.
                ``Q`` is the query sequence length, ``Hq`` is the number of
                query attention heads (``num_query_heads``), and ``C`` is the
                channels per head.
            key: The key tensor with shape ``[..., KV, Hkv, C]``.
                ``KV`` is the key/value sequence length and ``Hkv`` is the
                number of key/value heads (``num_key_value_heads``).
            value: The value tensor with shape ``[..., KV, Hkv, C]``.
            seqused_key_value: Valid key/value lengths with shape ``[...]`` and
                :external+torch:ref:`torch.int32 <dtype-doc>` dtype.
            attn_mask: Boolean attention mask with shape ``[..., Q, KV]``.
                Entries set to ``True`` participate in attention.

        Returns:
            Tensor with shape ``[..., Q, Hq, C]``.
        """
        if query.numel() == 0:
            return query

        if attn_mask is not None and seqused_key_value is not None:
            raise ValueError(
                "Cannot pass both `attn_mask` and `seqused_key_value`"
            )

        if (
            seqused_key_value is not None
            and seqused_key_value.dtype != torch.int32
        ):
            raise ValueError("`seqused_key_value` must have dtype torch.int32")
        if attn_mask is not None and attn_mask.dtype != torch.bool:
            raise ValueError("`attn_mask` must have dtype torch.bool")

        batch_shapes = [query.size()[:-3], key.size()[:-3], value.size()[:-3]]
        if seqused_key_value is not None:
            batch_shapes.append(seqused_key_value.size())
        if attn_mask is not None:
            batch_shapes.append(attn_mask.size()[:-2])
        batch_shape = torch.broadcast_shapes(*batch_shapes)

        if self.query_scaling is not None:
            if seqused_key_value is not None:
                key_len = seqused_key_value.unsqueeze(-1)
            elif attn_mask is not None and attn_mask.size(-1) > 1:
                key_len = attn_mask.sum(dim=-1)
            else:
                key_len = key.size(-3)
            query = self.query_scaling(query, key_len=key_len)

        # Broadcast and flatten batch dimensions => [B, S, H, C].
        query_size = query.size()[-3:]
        key_size = key.size()[-3:]
        value_size = value.size()[-3:]

        if key_size[0] == 0:  # No key/value pairs - abort early:
            return query.new_zeros(batch_shape + query_size)

        query = query.expand(batch_shape + query_size).reshape(-1, *query_size)
        key = key.expand(batch_shape + key_size).reshape(-1, *key_size)
        value = value.expand(batch_shape + value_size).reshape(-1, *value_size)

        if attn_mask is not None:
            attn_mask = attn_mask.expand(batch_shape + attn_mask.size()[-2:])
            attn_mask = attn_mask.reshape(-1, *attn_mask.size()[-2:])

        if seqused_key_value is not None:
            seqused_key_value = seqused_key_value.expand(batch_shape)
            seqused_key_value = seqused_key_value.reshape(-1).unsqueeze(-1)
            key_index = torch.arange(key.size(-3), device=key.device)
            attn_mask = key_index.unsqueeze(0) < seqused_key_value
            attn_mask = attn_mask.unsqueeze(-2).expand(-1, query.size(-3), -1)

        enable_gqa = False
        if query.size(-2) != key.size(-2):
            enable_gqa = True

        out = F.scaled_dot_product_attention(
            query=query.transpose(-3, -2),  # [B, Hq, Q, C],
            key=key.transpose(-3, -2),  # [B, Hkv, KV, C],
            value=value.transpose(-3, -2),  # [B, Hkv, KV, C],
            attn_mask=attn_mask.unsqueeze(-3)  # [B, 1, Q, KV]
            if attn_mask is not None
            else None,
            enable_gqa=enable_gqa,
            scale=self.scale,
        ).transpose(-3, -2)  # [B, Q, Hq, C]

        return out.view(batch_shape + out.size()[-3:])  # [..., Q, Hq, C]


class Attention(torch.nn.Module):
    r"""Multi-head attention layer with grouped-query attention support.

    This module owns the query, key, value, and output projections.
    It performs self-attention when ``key_value`` is omitted and
    cross-attention when ``key_value`` is given.

    Args:
        channels: The number of input and output channels.
        num_query_heads: The number of query attention heads.
            ``channels`` must be divisible by ``num_query_heads``.
        num_key_value_heads: The number of key/value attention heads.
            Setting this below ``num_query_heads`` enables grouped-query
            attention (GQA); setting it to ``1`` enables multi-query attention
            (MQA). Must divide ``num_query_heads``. Defaults to
            ``num_query_heads`` (standard multi-head attention).
        query_transform: Transformation applied to projected query heads before
            scaled dot-product attention.
        key_transform: Transformation applied to projected key heads before
            scaled dot-product attention.
        query_scaling: Query scaling module to scale projected query heads
            before scaled dot-product attention, *e.g.*, :class:`QASSMax`.
        scale: Scaling factor passed to
            :func:`torch.nn.functional.scaled_dot_product_attention`.
            ``None`` uses ``1 / sqrt(channels_per_head)``.
        bias: If set to ``False``, the module will not learn an additive bias.
        device: The device.
        dtype: The dtype.
    """

    def __init__(
        self,
        channels: int,
        num_query_heads: int,
        num_key_value_heads: int | None = None,
        query_transform: torch.nn.Module | None = None,
        key_transform: torch.nn.Module | None = None,
        query_scaling: QueryScaling | None = None,
        scale: float | None = None,
        bias: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if num_key_value_heads is None:
            num_key_value_heads = num_query_heads
        if channels % num_query_heads != 0:
            raise ValueError(
                f"`channels` ({channels}) must be divisible by "
                f"`num_query_heads` ({num_query_heads})"
            )

        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}

        self.num_query_heads = num_query_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = channels // num_query_heads
        # Query projection spans all channels; key/value span fewer heads.
        self.q_dim = num_query_heads * self.head_dim  # == channels
        self.kv_dim = num_key_value_heads * self.head_dim

        self.qkv_lin = Linear(
            channels, self.q_dim + 2 * self.kv_dim, bias=bias, **factory_kwargs
        )
        self.query_transform = query_transform
        self.key_transform = key_transform
        self.sdpa = SDPA(
            num_query_heads=num_query_heads,
            num_key_value_heads=num_key_value_heads,
            query_scaling=query_scaling,
            scale=scale,
        )
        self.out_lin = Linear(channels, channels, bias=bias, **factory_kwargs)

        torch.nn.init.zeros_(self.out_lin.weight)
        if self.out_lin.bias is not None:
            torch.nn.init.zeros_(self.out_lin.bias)

    @overload
    def forward(
        self,
        query: Tensor,
        key_value: Tensor | KVCacheEntry | None = None,
        seqused_key_value: Tensor | None = None,
        attn_mask: Tensor | None = None,
        *,
        return_key_value: Literal[False] = False,
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
    ) -> Tensor | tuple[Tensor, KVCacheEntry]: ...

    def forward(
        self,
        query: Tensor,  # [..., Q, C]
        key_value: Tensor | KVCacheEntry | None = None,  # [..., KV, C]
        seqused_key_value: Tensor | None = None,  # [...]
        attn_mask: Tensor | None = None,  # [..., Q, KV]
        *,
        return_key_value: bool = False,
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
                If omitted, ``query`` is used for self-attention.
            seqused_key_value: Valid key/value lengths with shape ``[...]`` and
                :external+torch:ref:`torch.int32 <dtype-doc>` dtype.
            attn_mask: Boolean attention mask with shape ``[..., Q, KV]``.
                Entries set to ``True`` participate in attention.
            return_key_value: Whether to return the computed key and value
                projections alongside the attention output.

        Returns:
            Tensor with shape ``[..., Q, C]`` when ``return_key_value`` is
            ``False``.
            Otherwise, a tuple of the output tensor and a
            :class:`~sdm.cache.KVCacheEntry`.
        """
        if isinstance(key_value, KVCacheEntry):
            query = F.linear(
                query,
                weight=self.qkv_lin.weight[: self.q_dim],
                bias=self.qkv_lin.bias[: self.q_dim]
                if self.qkv_lin.bias is not None
                else None,
            )
            if (
                key_value.key.dtype != query.dtype
                or key_value.value.dtype != query.dtype
            ):
                raise ValueError(
                    f"Key/value projections were cached under dtypes "
                    f"'{key_value.key.dtype}'/'{key_value.value.dtype}' but "
                    f"the query has dtype '{query.dtype}'"
                )
            key = key_value.key
            value = key_value.value
        elif key_value is None:
            query, key, value = self.qkv_lin(query).split(
                [self.q_dim, self.kv_dim, self.kv_dim], dim=-1
            )
        else:
            sections = [self.q_dim, 2 * self.kv_dim]
            q_weight, kv_weight = self.qkv_lin.weight.split(sections, dim=0)
            q_bias = kv_bias = None
            if self.qkv_lin.bias is not None:
                q_bias, kv_bias = self.qkv_lin.bias.split(sections, dim=0)
            query = F.linear(query, q_weight, q_bias)
            key, value = F.linear(key_value, kv_weight, kv_bias).chunk(2, -1)

        # [..., S, C] -> [..., S, H, C // H], with separate query/kv heads.
        query = query.unflatten(-1, [self.num_query_heads, self.head_dim])
        if not isinstance(key_value, KVCacheEntry):
            key = key.unflatten(-1, [self.num_key_value_heads, self.head_dim])
            value = value.unflatten(
                -1, [self.num_key_value_heads, self.head_dim]
            )

        if self.query_transform is not None:
            query = self.query_transform(query)
        if (
            not isinstance(key_value, KVCacheEntry)
            and self.key_transform is not None
        ):
            key = self.key_transform(key)

        out = self.sdpa(
            query=query,  # [..., Q, Hq, C // Hq]
            key=key,  # [..., KV, Hkv, C // Hq]
            value=value,  # [..., KV, Hkv, C // Hq]
            seqused_key_value=seqused_key_value,  # [...]
            attn_mask=attn_mask,  # [..., Q, KV]
        )  # [..., Q, Hq, C // Hq]

        out = out.flatten(-2, -1)  # [..., Q, C]
        out = self.out_lin(out)  # [..., Q, C]
        if return_key_value:
            # CUDA autocast runs RMSNorm in fp32, so cast explicitly before
            # caching: https://github.com/pytorch/pytorch/blob/v2.13.0/aten/src/ATen/autocast_mode.h#L875
            return out, KVCacheEntry(
                key=key.to(value.dtype),
                value=value,
            )
        return out


class TransformerBlock(torch.nn.Module):
    r"""Transformer block with normalization and feedforward residual modules.

    Args:
        channels: The number of input and output channels.
        num_query_heads: The number of query attention heads.
        mlp: Feedforward module applied after the attention residual.
        num_key_value_heads: The number of key/value attention heads.
            Defaults to ``num_query_heads`` (standard multi-head attention).
        query_norm: Normalization applied to query inputs before attention.
        key_value_norm: Normalization applied to key/value inputs before
            attention.
        post_attn_norm: Normalization applied to the attention output before
            its residual addition.
        query_transform: Transformation applied to projected query heads before
            scaled dot-product attention.
        key_transform: Transformation applied to projected key heads before
            scaled dot-product attention.
        query_scaling: Query scaling module to scale projected query heads
            before scaled dot-product attention, *e.g.*, :class:`QASSMax`.
        scale: Scaling factor passed to
            :func:`torch.nn.functional.scaled_dot_product_attention`.
            ``None`` uses ``1 / sqrt(channels_per_head)``.
        bias: If set to ``False``, the module will not learn an additive bias.
        device: The device.
        dtype: The dtype.
    """

    def __init__(
        self,
        channels: int,
        num_query_heads: int,
        mlp: torch.nn.Module,
        num_key_value_heads: int | None = None,
        query_norm: torch.nn.Module | None = None,
        key_value_norm: torch.nn.Module | None = None,
        post_attn_norm: torch.nn.Module | None = None,
        query_transform: torch.nn.Module | None = None,
        key_transform: torch.nn.Module | None = None,
        query_scaling: QueryScaling | None = None,
        scale: float | None = None,
        bias: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}

        self.mlp = mlp
        self.query_norm = query_norm
        self.key_value_norm = key_value_norm
        self.post_attn_norm = post_attn_norm

        self.attn = Attention(
            channels=channels,
            num_query_heads=num_query_heads,
            num_key_value_heads=num_key_value_heads,
            query_transform=query_transform,
            key_transform=key_transform,
            query_scaling=query_scaling,
            scale=scale,
            bias=bias,
            **factory_kwargs,
        )

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
        attn_mask: Tensor | None = None,  # [..., Q, KV]
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
                If omitted, ``query`` is used for self-attention.
            seqused_key_value: Valid key/value lengths with shape ``[...]`` and
                :external+torch:ref:`torch.int32 <dtype-doc>` dtype.
            attn_mask: Boolean attention mask with shape ``[..., Q, KV]``.
                Entries set to ``True`` participate in attention.
            return_key_value: Whether to return the computed key and value
                projections alongside the block output.
            batch_size_limit: Maximum number of batch elements processed at
                once.
            out: The output tensor.

        Returns:
            Tensor with shape ``[..., Q, C]`` when ``return_key_value`` is
            ``False``.
            Otherwise, a tuple of the output tensor and a
            :class:`~sdm.cache.KVCacheEntry`.
        """
        if out is not None and torch.is_grad_enabled():
            raise RuntimeError(
                "'out' is only supported when gradients are disabled"
            )
        if return_key_value and isinstance(key_value, KVCacheEntry):
            raise ValueError(
                "'return_key_value=True' is not supported when 'key_value' is "
                "already cached"
            )

        if torch.is_grad_enabled() or torch.compiler.is_compiling():
            return self._forward(
                query=query,
                key_value=key_value,
                seqused_key_value=seqused_key_value,
                attn_mask=attn_mask,
                return_key_value=return_key_value,
                out=out,
            )

        if batch_size_limit == "auto":
            key_value_length: int | None = None
            if isinstance(key_value, Tensor):
                key_value_length = key_value.size(-2)
            elif isinstance(key_value, KVCacheEntry):
                key_value_length = key_value.key.size(-3)

            batch_size_limit = self.auto_batch_size_limit(
                device=query.device,
                element_size=torch.get_autocast_dtype(
                    query.device.type
                ).itemsize
                if torch.is_autocast_enabled(query.device.type)
                else query.element_size(),
                query_length=query.size(-2),
                key_value_length=key_value_length,
            )

        batch_size_limit = min(batch_size_limit or 65_535, 65_535)

        batch_shape = _batch_shape(
            query=query,
            key_value=key_value,
            seqused_key_value=seqused_key_value,
            attn_mask=attn_mask,
        )
        batch_size = math.prod(batch_shape)

        disable_chunking = False
        if batch_size <= batch_size_limit or query.size(-2) == 0:
            disable_chunking = True
        if return_key_value:
            assert not isinstance(key_value, KVCacheEntry)
            if batch_shape != (
                query.size()[:-2]
                if key_value is None
                else key_value.size()[:-2]
            ):
                disable_chunking = True

        if disable_chunking:
            return self._forward(
                query=query,
                key_value=key_value,
                seqused_key_value=seqused_key_value,
                attn_mask=attn_mask,
                return_key_value=return_key_value,
                out=out,
            )

        flat_out: Tensor | None = None
        flat_key: Tensor | None = None
        flat_value: Tensor | None = None

        if out is not None and (out.dim() <= 3 or out.is_contiguous()):
            flat_out = out.view(batch_size, *query.size()[-2:])

        for start in range(0, batch_size, batch_size_limit):
            end = min(start + batch_size_limit, batch_size)

            result = self._forward(
                query=_chunk(query, batch_shape, 2, start, end),
                key_value=_chunk(key_value, batch_shape, 2, start, end),
                seqused_key_value=_chunk(
                    seqused_key_value,
                    batch_shape,
                    trailing_dims=0,
                    start=start,
                    end=end,
                ),
                attn_mask=_chunk(
                    attn_mask,
                    batch_shape,
                    trailing_dims=2,
                    start=start,
                    end=end,
                ),
                return_key_value=return_key_value,
                out=flat_out[start:end] if flat_out is not None else None,
            )

            if isinstance(result, tuple):
                chunk, chunk_kv = result

                if flat_key is None:
                    flat_key = chunk_kv.key.new_empty(
                        batch_size, *chunk_kv.key.size()[-3:]
                    )
                if flat_value is None:
                    flat_value = chunk_kv.value.new_empty(
                        batch_size, *chunk_kv.value.size()[-3:]
                    )

                flat_key[start:end] = chunk_kv.key
                flat_value[start:end] = chunk_kv.value

                del chunk_kv

            else:
                chunk = result

            if flat_out is None and out is not None:
                flat_index = torch.arange(start, end, device=out.device)
                batch_indices: list[Tensor] = []
                for size in reversed(batch_shape):
                    batch_indices.append(flat_index % size)
                    flat_index = flat_index // size
                out[tuple(reversed(batch_indices))] = chunk
            elif flat_out is None:
                flat_out = chunk.new_empty((batch_size, *query.size()[-2:]))
                flat_out[start:end] = chunk

            del chunk, result

        if out is None:
            assert flat_out is not None
            out = flat_out.view(*batch_shape, *query.size()[-2:])

        if not return_key_value:
            return out

        assert flat_key is not None
        assert flat_value is not None

        return out, KVCacheEntry(
            key=flat_key.view(*batch_shape, *flat_key.size()[-3:]),
            value=flat_value.view(*batch_shape, *flat_value.size()[-3:]),
        )

    def _forward(
        self,
        query: Tensor,  # [..., Q, C]
        key_value: Tensor | KVCacheEntry | None = None,  # [..., KV, C]
        seqused_key_value: Tensor | None = None,  # [...]
        attn_mask: Tensor | None = None,  # [..., Q, KV]
        *,
        return_key_value: bool = False,
        out: Tensor | None = None,
    ) -> Tensor | tuple[Tensor, KVCacheEntry]:  # [..., Q, C]

        if self.key_value_norm is None:
            pass
        elif key_value is None and self.key_value_norm is not self.query_norm:
            key_value = self.key_value_norm(query)
        elif isinstance(key_value, Tensor):
            key_value = self.key_value_norm(key_value)

        result = self.attn(
            query=query if self.query_norm is None else self.query_norm(query),
            key_value=key_value,
            seqused_key_value=seqused_key_value,
            attn_mask=attn_mask,
            return_key_value=return_key_value,
        )
        del key_value

        if return_key_value:
            attn_out, kv = result
        else:
            attn_out = result

        if self.post_attn_norm is not None:
            attn_out = self.post_attn_norm(attn_out)

        if (
            out is not None
            and torch.compiler.is_compiling()
            and not out.is_contiguous()
        ):
            tmp = attn_out + query
            out.copy_(tmp + self.mlp(tmp))
        else:
            tmp = torch.add(attn_out, query, out=out)
            out = torch.add(tmp, self.mlp(tmp), out=out)

        return (out, kv) if return_key_value else out

    def peak_bytes_per_example(
        self,
        element_size: int,
        query_length: int,
        key_value_length: int | None = None,
    ) -> int:
        r""":meta private:"""  # noqa: D415
        return 0

    def auto_batch_size_limit(
        self,
        device: torch.device,
        element_size: int,
        query_length: int,
        key_value_length: int | None = None,
    ) -> int:
        r""":meta private:"""  # noqa: D415
        if device.type != "cuda":
            return 65_535
        bytes_per_example = self.peak_bytes_per_example(
            element_size=element_size,
            query_length=query_length,
            key_value_length=key_value_length,
        )
        limit = chunk_memory_limit(device) // max(bytes_per_example, 1)
        return min(max(limit, 1), 65_535)


def _batch_shape(
    query: Tensor,
    key_value: Tensor | KVCacheEntry | None,
    seqused_key_value: Tensor | None,
    attn_mask: Tensor | None,
) -> torch.Size:
    shapes = [query.size()[:-2]]
    if isinstance(key_value, Tensor):
        shapes.append(key_value.size()[:-2])
    elif isinstance(key_value, KVCacheEntry):
        shapes += [key_value.key.size()[:-3], key_value.value.size()[:-3]]
    if seqused_key_value is not None:
        shapes.append(seqused_key_value.size())
    if attn_mask is not None:
        shapes.append(attn_mask.size()[:-2])
    return torch.broadcast_shapes(*shapes)


@overload
def _chunk(
    tensor: Tensor,
    batch_shape: torch.Size,
    trailing_dims: int,
    start: int,
    end: int,
) -> Tensor: ...


@overload
def _chunk(
    tensor: KVCacheEntry,
    batch_shape: torch.Size,
    trailing_dims: int,
    start: int,
    end: int,
) -> KVCacheEntry: ...


@overload
def _chunk(
    tensor: None,
    batch_shape: torch.Size,
    trailing_dims: int,
    start: int,
    end: int,
) -> None: ...


def _chunk(
    tensor: Tensor | KVCacheEntry | None,
    batch_shape: torch.Size,
    trailing_dims: int,
    start: int,
    end: int,
) -> Tensor | KVCacheEntry | None:

    if tensor is None:
        return None

    if isinstance(tensor, KVCacheEntry):
        return KVCacheEntry(
            key=_chunk(tensor.key, batch_shape, 3, start, end),
            value=_chunk(tensor.value, batch_shape, 3, start, end),
        )

    trailing_shape = tensor.size()[-trailing_dims:] if trailing_dims else ()
    tensor = tensor.expand(batch_shape + trailing_shape)
    if len(batch_shape) == 1:
        return tensor[start:end]

    flat_index = torch.arange(start, end, device=tensor.device)
    batch_indices: list[Tensor] = []
    for size in reversed(batch_shape):
        batch_indices.append(flat_index % size)
        flat_index = flat_index // size
    return tensor[tuple(reversed(batch_indices))]
