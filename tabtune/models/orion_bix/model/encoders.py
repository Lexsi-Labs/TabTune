from __future__ import annotations
from typing import Optional
from torch import nn, Tensor

from .layers import LinearAttentionBlock,InducedSelfAttentionBlock,MultiheadAttentionBlock
from .rope import RotaryEmbedding

class Encoder(nn.Module):
    """
    Stack of linear attention blocks for encoding input sequences.

    Args:
        num_blocks (int): Number of attention blocks.
        d_model (int): Embedding dimension.
        nhead (int): Number of attention heads.
        dim_feedforward (int): Feedforward network dimension.
        dropout (float): Dropout rate.
        activation (str): Activation function.
        norm_first (bool): Whether to use pre-norm.
        use_rope (bool): Whether to use rotary positional encoding.
        rope_base (int): Base for rotary encoding.
        feature_map (str): Feature map type ("elu", "identity", "hedgehog").
    """

    def __init__(
        self,
        num_blocks: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.0,
        activation: str = "gelu",
        norm_first: bool = True,
        use_rope: bool = False,
        rope_base: int = 100000,
        attention_type: str = "linear",
        feature_map: str = "elu",  # "elu", "identity", "hedgehog"
        debug: bool = False,
    ):
        super().__init__()
        self.attention_type = attention_type
        self.debug = debug
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")

        if self.attention_type == "linear":
            self.blocks = nn.ModuleList(
                [
                    LinearAttentionBlock(
                        d_model=d_model,
                        nhead=nhead,
                        dim_feedforward=dim_feedforward,
                        dropout=dropout,
                        activation=activation,
                        feature_map=feature_map,
                    )
                    for _ in range(num_blocks)
                ]
            )

            self.rope = RotaryEmbedding(dim=d_model // nhead, theta=rope_base, use_xpos=True) if use_rope else None
            if self.rope and self.debug:
                print(f"[DEBUG] RoPE initialized with use_xpos: {self.rope.use_xpos}")
        else:
            self.blocks = nn.ModuleList(
            [
                MultiheadAttentionBlock(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation=activation,
                    norm_first=norm_first,
                )
                for _ in range(num_blocks)
            ]
        )

            self.rope = RotaryEmbedding(dim=d_model // nhead, theta=rope_base) if use_rope else None

    def forward(
        self,
        src: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor | int] = None,
    ) -> Tensor:
        """
        Forward pass through the encoder stack.

        Args:
            src (Tensor): Input tensor.
            key_padding_mask (Optional[Tensor]): Optional mask for padding.
            attn_mask (Optional[Tensor|int]): Optional attention mask or train size.
        Returns:
            Tensor: Encoded output.
        """
        if self.debug:
            print(f"[DEBUG] Encoder input:")
            print(f"  src: {src.shape}")
            print(f"  key_padding_mask: {key_padding_mask}")
            print(f"  attn_mask: {attn_mask}")
            print(f"  rope: {self.rope}")
            if self.rope:
                print(f"  rope.use_xpos: {self.rope.use_xpos}")
                print(f"  rope.freqs.shape: {self.rope.freqs.shape}")
        
        out = src
        if self.attention_type == "linear":
            for i, block in enumerate(self.blocks):
                out = block(src=out, train_size=attn_mask, rope=self.rope)
        else:
            for block in self.blocks:
                out = block(q=out, key_padding_mask=key_padding_mask, attn_mask=attn_mask, rope=self.rope)

        return out


class SetTransformer(nn.Module):
    """
    Stack of linear attention blocks for set-structured input.

    Args:
        num_blocks (int): Number of attention blocks.
        d_model (int): Embedding dimension.
        nhead (int): Number of attention heads.
        dim_feedforward (int): Feedforward network dimension.
        dropout (float): Dropout rate.
        activation (str): Activation function.
        norm_first (bool): Whether to use pre-norm.
        feature_map (str): Feature map type ("elu", "identity", "hedgehog").
    """

    def __init__(
        self,
        num_blocks: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_inds: int = 16,
        dropout: float = 0.0,
        activation: str = "gelu",
        norm_first: bool = True,
        attention_type: str = "linear",
        feature_map: str = "elu",  # "elu", "identity", "hedgehog"
        debug: bool = False,
    ):
        super().__init__()
        self.attention_type = attention_type

        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")

        if self.attention_type == "linear":
            self.blocks = nn.ModuleList(
                [
                    LinearAttentionBlock(
                        d_model=d_model,
                        nhead=nhead,
                        dim_feedforward=dim_feedforward,
                        dropout=dropout,
                        activation=activation,
                        feature_map=feature_map,
                        debug=debug,
                    )
                    for _ in range(num_blocks)
                ]
            )
        else:
            self.blocks = nn.ModuleList(
            [
                InducedSelfAttentionBlock(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    num_inds=num_inds,
                    dropout=dropout,
                    activation=activation,
                    norm_first=norm_first,
                )
                for _ in range(num_blocks)
            ]
            )


    def forward(self, src: Tensor, train_size: Optional[int] = None) -> Tensor:
        """
        Forward pass through the set transformer stack.

        Args:
            src (Tensor): Input tensor.
            train_size (Optional[int]): Optional train size for masking.
        Returns:
            Tensor: Encoded output.
        """
        out = src
        for block in self.blocks:
            out = block(out, train_size)

        return out