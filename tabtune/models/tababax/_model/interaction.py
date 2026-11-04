# This file is based on TabICL v2 and has been modified by TabAbax.
# https://github.com/soda-inria/tabicl
#
# Original copyright:
# Copyright (c) 2025, Soda team @ Inria
#
# TabAbax modifications:
# Copyright (c) 2026 TabAbax
#
# The original TabICL-derived portions remain under BSD-3-Clause.
# See NOTICE and LICENSES/BSD-3-Clause.txt.

from __future__ import annotations

from typing import Optional
from functools import partial
from collections import OrderedDict

import torch
from torch import nn, Tensor
from torch.utils.checkpoint import checkpoint

from .encoders import Encoder
from .inference import InferenceManager
from .inference_config import MgrConfig, InferenceConfig


class RowInteraction(nn.Module):
    """Context-aware row-wise interaction.

    This module captures interactions between features within each row using a transformer
    encoder with rotary positional encoding. It prepends learnable class tokens to the
    learned feature embeddings and uses these tokens to aggregate information.

    Parameters
    ----------
    embed_dim : int
        Embedding dimension.

    num_blocks : int
        Number of blocks used in the encoder.

    nhead : int
        Number of attention heads of the encoder.

    dim_feedforward : int
        Dimension of the feedforward network of the encoder.

    num_cls : int, default=4
        Number of learnable CLS tokens to prepend to the feature embeddings. The outputs
        of these CLS tokens are concatenated for the final representation per row.

    rope_base : float, default=100000
        Base scaling factor for rotary position encoding.

    rope_interleaved : bool, default=True
        If True, uses interleaved rotation where dimension pairs are (0,1), (2,3), etc.
        If False, uses non-interleaved rotation where the embedding is split into
        first half [0:d//2] and second half [d//2:d].

    dropout : float, default=0.0
        Dropout probability used in the encoder.

    activation : str or unary callable, default="gelu"
        The activation function used in the feedforward network, can be
        either string ("relu" or "gelu") or unary callable.

    norm_first : bool, default=True
        If True, uses pre-norm architecture (LayerNorm before attention and feedforward).

    bias_free_ln : bool, default=False
        If True, removes bias from all LayerNorm layers.

    recompute : bool, default=False
        If True, uses gradient checkpointing to save memory at the cost of additional computation.
    """

    def __init__(
        self,
        embed_dim: int,
        num_blocks: int,
        nhead: int,
        dim_feedforward: int,
        num_cls: int = 4,
        rope_base: float = 100000,
        rope_interleaved: bool = True,
        dropout: float = 0.0,
        activation: str | callable = "gelu",
        norm_first: bool = True,
        bias_free_ln: bool = False,
        zero_init: bool = True,
        recompute: bool = False,
        output_full_sequence: bool = False,
        inject_cls_tokens: bool = True,
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.num_blocks = num_blocks
        self.num_cls = num_cls
        self.norm_first = norm_first
        self.recompute = recompute
        self.output_full_sequence = output_full_sequence
        self.inject_cls_tokens = inject_cls_tokens

        self.tf_row = Encoder(
            num_blocks=num_blocks,
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            bias_free_ln=bias_free_ln,
            use_rope=True,
            rope_base=rope_base,
            rope_interleaved=rope_interleaved,
            zero_init=zero_init,
            recompute=recompute,
        )

        if inject_cls_tokens:
            self.cls_tokens = nn.Parameter(
                torch.empty(num_cls, embed_dim)
            )
            nn.init.trunc_normal_(self.cls_tokens, std=0.02)
        else:
            self.register_parameter("cls_tokens", None)

        self.out_ln = (
            nn.LayerNorm(
                embed_dim,
                bias=not bias_free_ln,
            )
            if norm_first
            else nn.Identity()
        )

        # 暂时不使用，但保留给后续分块、offload 等推理优化。
        # 第一轮输出保留 token 维，第二轮输出已经合并 CLS 维。
        self.inference_mgr = InferenceManager(
            enc_name="tf_row",
            out_dim=(
                embed_dim
                if output_full_sequence
                else embed_dim * self.num_cls
            ),
            out_no_seq=not output_full_sequence,
        )


    def _aggregate_embeddings(
        self,
        embeddings: Tensor,
        key_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Apply row attention.

        When output_full_sequence=True:
            (B, T, H+C, E) -> (B, T, H+C, E)

        When output_full_sequence=False:
            (B, T, H+C, E) -> (B, T, C*E)
        """

        rope = self.tf_row.rope

        # 前面的 block 始终处理完整序列
        if self.recompute:
            for block in self.tf_row.blocks[:-1]:
                embeddings = checkpoint(
                    partial(
                        block,
                        key_padding_mask=key_mask,
                        rope=rope,
                    ),
                    embeddings,
                    use_reentrant=False,
                )
        else:
            for block in self.tf_row.blocks[:-1]:
                embeddings = block(
                    embeddings,
                    key_padding_mask=key_mask,
                    rope=rope,
                )

        last_block = self.tf_row.blocks[-1]

        # 第一轮 Row Attention：
        # 最后一个 block 也处理并返回全部 CLS + feature tokens
        if self.output_full_sequence:
            if self.recompute:
                outputs = checkpoint(
                    partial(
                        last_block,
                        key_padding_mask=key_mask,
                        rope=rope,
                    ),
                    embeddings,
                    use_reentrant=False,
                )
            else:
                outputs = last_block(
                    embeddings,
                    key_padding_mask=key_mask,
                    rope=rope,
                )

            return self.out_ln(outputs)

        # 第二轮 Row Attention：
        # 最后一个 block 只计算 CLS query
        if self.recompute:
            outputs = checkpoint(
                lambda emb: last_block(
                    q=emb[..., : self.num_cls, :],
                    k=emb,
                    v=emb,
                    key_padding_mask=key_mask,
                    rope=rope,
                ),
                embeddings,
                use_reentrant=False,
            )
        else:
            outputs = last_block(
                q=embeddings[..., : self.num_cls, :],
                k=embeddings,
                v=embeddings,
                key_padding_mask=key_mask,
                rope=rope,
            )

        del embeddings

        outputs = self.out_ln(outputs)

        return outputs.flatten(-2)

    def _train_forward(
        self,
        embeddings: Tensor,
        d: Optional[Tensor] = None,
    ) -> Tensor:
        return self._direct_forward(
            embeddings,
            d,
        )

    def _inference_forward(
        self,
        embeddings: Tensor,
        d: Optional[Tensor] = None,
        mgr_config: Optional[MgrConfig] = None,
    ) -> Tensor:
        """Inference through the official InferenceManager path."""

        if mgr_config is None:
            mgr_config = InferenceConfig().ROW_CONFIG

        self.inference_mgr.configure(**mgr_config)

        B, T, HC, E = embeddings.shape

        if E != self.embed_dim:
            raise ValueError(
                f"Expected embedding dimension {self.embed_dim}, "
                f"but got {E}"
            )

        if HC < self.num_cls:
            raise ValueError(
                f"Expected at least {self.num_cls} token positions, "
                f"but got {HC}"
            )

        # 第一轮 Row 必须在推理时注入 CLS token。
        # 第二轮 Row 的 inject_cls_tokens=False，
        # 必须保留第一轮产生的 CLS 表示。
        if self.inject_cls_tokens:
            cls_tokens = self.cls_tokens.expand(
                B,
                T,
                self.num_cls,
                self.embed_dim,
            )

            # 不使用原地写入，避免输入是 leaf tensor 或被前一阶段复用时
            # 产生 autograd / aliasing 问题。
            # embeddings = torch.cat(
            #     [
            #         cls_tokens.to(
            #             device=embeddings.device,
            #             dtype=embeddings.dtype,
            #         ),
            #         embeddings[:, :, self.num_cls:],
            #     ],
            #     dim=2,
            # )
            embeddings[:, :, : self.num_cls] = cls_tokens.to(
                device=embeddings.device,
                dtype=embeddings.dtype,
            )

        # 与官方推理路径保持一致：
        # 推理阶段当前不使用 d 构造 feature padding mask。
        _ = d

        return self.inference_mgr(
            self._aggregate_embeddings,
            inputs=OrderedDict(
                [
                    ("embeddings", embeddings),
                ]
            ),
        )

    def forward(
        self,
        embeddings: Tensor,
        d: Optional[Tensor] = None,
        mgr_config: Optional[MgrConfig] = None,
    ) -> Tensor:
        """Transform feature embeddings."""

        if self.training:
            return self._train_forward(
                embeddings,
                d,
            )

        return self._inference_forward(
            embeddings,
            d,
            mgr_config,
        )

    def _direct_forward(
        self,
        embeddings: Tensor,
        d: Optional[Tensor] = None,
    ) -> Tensor:
        """Reference forward shared by training and basic inference."""

        B, T, HC, E = embeddings.shape
        device = embeddings.device

        if E != self.embed_dim:
            raise ValueError(
                f"Expected embedding dimension {self.embed_dim}, "
                f"but got {E}"
            )

        if HC < self.num_cls:
            raise ValueError(
                f"Expected at least {self.num_cls} token positions, "
                f"but got {HC}"
            )

        # 只允许第一轮写入 CLS。
        if self.inject_cls_tokens:
            cls_tokens = self.cls_tokens.expand(
                B,
                T,
                self.num_cls,
                self.embed_dim,
            )

            # 替换 ColEmbedding 预留的 CLS 空槽。
            #
            # 不使用原来的原地赋值：
            # embeddings[:, :, :self.num_cls] = cls_tokens
            #
            # torch.cat 对 autograd 更安全，输入即使是 requires_grad
            # 的叶子张量也不会报原地修改错误。
            embeddings = torch.cat(
                [
                    cls_tokens.to(device),
                    embeddings[:, :, self.num_cls :],
                ],
                dim=2,
            )

        # 屏蔽批次中不存在的 padding features。
        if d is None:
            key_mask = None
        else:
            d_with_cls = d + self.num_cls

            indices = torch.arange(
                HC,
                device=device,
            ).view(1, 1, HC).expand(B, T, HC)

            key_mask = indices >= d_with_cls.view(B, 1, 1)

        return self._aggregate_embeddings(
            embeddings,
            key_mask,
        )
