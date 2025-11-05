# interaction.py
from __future__ import annotations

from typing import Optional, Tuple
from collections import OrderedDict
import math

import torch
from torch import nn, Tensor

from .encoders import Encoder
from .inference import InferenceManager
from .inference_config import MgrConfig


# ------------------------- Sparse mask builder ------------------------- #

def _build_block_sparse_mask(
    seq_len: int,
    num_special: int,
    window: int,
    num_random: int = 0,
    device=None,
    dtype=torch.float32,
    return_bool: bool = False,
) -> Tensor:
    """
    Build additive attention mask (0 allow, -inf disallow):
      - first `num_special` tokens (CLS+GLOBAL) are fully connected in both directions
      - others use sliding window ±window
      - optional BigBird-style random links per non-special query
    """
    L = seq_len
    mask = torch.full((L, L), float("-inf"), device=device, dtype=dtype)

    if num_special > 0:
        mask[:num_special, :] = 0.0
        mask[:, :num_special] = 0.0

    if L > num_special and window >= 0:
        idx = torch.arange(L, device=device)
        dist = (idx.unsqueeze(1) - idx.unsqueeze(0)).abs()
        local = (dist <= window).to(mask.dtype) * 0.0
        mask = torch.where(mask.isfinite(), mask, local + float("-inf"))

    if num_random > 0 and L > num_special + 1:
        rng_idx = torch.arange(num_special, L, device=device)
        for i in rng_idx:
            #choices = (i + 1 + torch.arange(num_random, device=device)) % (L - num_special) + num_special
            choices = torch.randperm(L - num_special, device=device)[:num_random] + num_special
            mask[i, choices] = 0.0

    mask.fill_diagonal_(0.0)
    if return_bool:
        return ~mask.isfinite()
    return mask


# --------------------- Contiguous (mean) grouping ---------------------- #

def _group_features_avg(
    feats: Tensor, valid_counts: Optional[Tensor], group: int
) -> Tuple[Tensor, Optional[Tensor]]:
    """
    Contiguous-by-index grouping with (weighted) mean pooling.

    feats: (B, T, H, E) feature tokens (no specials)
    valid_counts: (B,) number of valid features per table, or None
    group: int grouping size (1 = no grouping)

    Returns:
      grouped: (B, T, Hg, E) where Hg = ceil(H/group)
      None (key mask handled upstream for uniformity)
    """
    B, T, H, E = feats.shape
    if group == 1:
        return feats, None

    pad = (-H) % group
    if pad:
        feats = torch.cat([feats, feats.new_zeros(B, T, pad, E)], dim=2)

    H_pad = feats.shape[2]
    Hg = H_pad // group
    feats = feats.view(B, T, Hg, group, E)

    if valid_counts is None:
        grouped = feats.mean(dim=3)
        return grouped, None
    else:
        device = feats.device
        d = valid_counts.clamp(max=H)
        idx = torch.arange(H_pad, device=feats.device)
        valid_feat = (idx[None, None, :, None] < d[:, None, None, None]).to(feats.dtype)
        valid_feat = valid_feat.view(B, 1, Hg, group, 1)
        weighted = feats * valid_feat
        denom = valid_feat.sum(dim=3).clamp(min=1e-6)
        grouped = weighted.sum(dim=3) / denom
        return grouped, None


# --------------------------- PMA grouping ------------------------------ #

class PMAGroup(nn.Module):
    """Soft grouping with learnable queries (Set-Transformer style PMA)."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.seed = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

    def _make_queries(self, BT: int, K: int, device) -> Tensor:
        E = self.embed_dim
        q = self.seed.expand(BT, K, E).to(device)
        pos = torch.arange(K, device=device).view(1, K, 1)
        omega = torch.exp(torch.arange(0, E, 2, device=device).float() * (-math.log(10000.0) / E))
        pe = torch.zeros(BT, K, E, device=device)
        pe[..., 0::2] = torch.sin(pos * omega)
        pe[..., 1::2] = torch.cos(pos * omega)
        return q + pe

    def forward(self, feats: Tensor, valid_counts: Optional[Tensor], K: int) -> Tensor:
        B, T, H, E = feats.shape
        device = feats.device
        BT = B * T

        x = feats.reshape(BT, H, E)
        Q = self.q_proj(self._make_queries(BT, K, device))
        Kmat = self.k_proj(x)
        Vmat = self.v_proj(x)

        if valid_counts is not None:
            d = valid_counts.clamp(max=H)
            idx = torch.arange(H, device=device).view(1, 1, H)
            key_mask = (idx >= d.view(B, 1, 1)).expand(B, T, H).reshape(BT, 1, H)
        else:
            key_mask = None

        attn_scores = (Q @ Kmat.transpose(1, 2)) / math.sqrt(E)
        if key_mask is not None:
            attn_scores = attn_scores.masked_fill(key_mask, float("-inf"))
        attn = torch.softmax(attn_scores, dim=-1)
        grouped = attn @ Vmat
        return grouped.view(B, T, K, E)


# --------------------------- RowInteraction --------------------------- #

class RowInteraction(nn.Module):
    """
    Hierarchical, block-sparse row-wise interaction with RoPE, GLOBAL tokens,
    and configurable grouping: "contiguous" (mean) or "pma" (learned).
    """

    def __init__(
        self,
        embed_dim: int,
        num_blocks: int,
        nhead: int,
        dim_feedforward: int,
        num_cls: int = 4,
        num_global: int = 2,
        rope_base: float = 100000,
        dropout: float = 0.0,
        activation: str | callable = "gelu",
        norm_first: bool = True,
        # multi-scale + sparsity knobs
        scales: Tuple[int, ...] = (1, 4, 8),
        window: int = 4,
        num_random: int = 0,
        # grouping
        group_mode: str = "contiguous",  # or "pma"
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_cls = num_cls
        self.num_global = num_global
        self.num_special = num_cls + num_global
        self.norm_first = norm_first
        self.scales = tuple(sorted(set(scales)))
        self.window = int(window)
        self.num_random = int(num_random)
        self.group_mode = group_mode.lower()
        if self.group_mode not in {"contiguous", "pma"}:
            raise ValueError("group_mode must be 'contiguous' or 'pma'")
            
        if any(s < 1 for s in self.scales):
            raise ValueError("All scales must be >= 1")
        if len(self.scales) == 0:
            raise ValueError("At least one scale must be specified")

        blocks_per_scale = max(1, math.ceil(num_blocks / max(1, len(self.scales))))
        self.scale_encoders = nn.ModuleList(
            [
                Encoder(
                    num_blocks=blocks_per_scale,
                    d_model=embed_dim,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation=activation,
                    norm_first=norm_first,
                    use_rope=True,
                    rope_base=rope_base,
                )
                for _ in self.scales
            ]
        )

        self.cls_tokens = nn.Parameter(torch.empty(num_cls, embed_dim))
        nn.init.trunc_normal_(self.cls_tokens, std=0.02)
        self.global_tokens = nn.Parameter(torch.empty(num_global, embed_dim))
        nn.init.trunc_normal_(self.global_tokens, std=0.02)

        self.pma = PMAGroup(embed_dim)  # used only if group_mode == "pma"
        self.out_ln = nn.LayerNorm(embed_dim) if norm_first else nn.Identity()
        self.inference_mgr = InferenceManager(enc_name="tf_row", out_dim=embed_dim * self.num_cls, out_no_seq=True)

    # ---- internal helpers ---- #

    def _group_per_scale(self, feats: Tensor, d: Optional[Tensor], group_size: int) -> Tensor:
        B, T, H, E = feats.shape
        if self.group_mode == "contiguous":
            grouped, _ = _group_features_avg(feats, d, group=group_size)
            return grouped
        else:
            K = int(math.ceil(H / max(1, group_size)))
            return self.pma(feats, d, K=K)

    def _run_one_scale(
        self,
        feats: Tensor,               # (B, T, H, E)
        specials: Tensor,            # (B, T, num_special, E)
        d: Optional[Tensor],         # (B,)
        encoder: Encoder,
        group_size: int,
    ) -> Tensor:
        device, dtype = feats.device, feats.dtype
        grouped = self._group_per_scale(feats, d, group_size=group_size)  # (B, T, K, E)
        K = grouped.shape[2]

        seq = torch.cat([specials, grouped], dim=2)  # (B, T, num_special + K, E)

        key_mask = None
        if d is not None:
            d_groups = (d.add(max(1, group_size) - 1) // max(1, group_size)).clamp(min=0)
            total_len = self.num_special + K
            idx = torch.arange(total_len, device=device).view(1, 1, total_len)
            per_table = self.num_special + d_groups.view(B:=feats.shape[0], 1, 1)
            key_mask = idx >= per_table
            key_mask = key_mask.expand(B, feats.shape[1], total_len)

        L = seq.shape[2]
        attn_mask_2d = _build_block_sparse_mask(
            seq_len=L,
            num_special=self.num_special,
            window=self.window,
            num_random=self.num_random,
            device=device,
            dtype=dtype,
            return_bool=True, 
        )
        
        if key_mask is not None and key_mask.dtype is not torch.bool:
            key_mask = key_mask.bool()

        out = encoder(seq, key_padding_mask=key_mask, attn_mask=attn_mask_2d)  # (B, T, L, E)
        return out[..., : self.num_cls, :]  # (B, T, num_cls, E)

    def _aggregate_embeddings(self, embeddings: Tensor, d: Optional[Tensor] = None) -> Tensor:
        B, T, HC, E = embeddings.shape
        device = embeddings.device

        specials = torch.cat(
            [
                self.cls_tokens.expand(B, T, self.num_cls, E),
                self.global_tokens.expand(B, T, self.num_global, E),
            ],
            dim=2,
        ).to(device)

        feats = embeddings[..., self.num_special :, :]  # (B, T, H, E)

        cls_per_scale = []
        for encoder, g in zip(self.scale_encoders, self.scales):
            cls_per_scale.append(self._run_one_scale(feats, specials, d, encoder, group_size=g))
        cls_outputs = torch.stack(cls_per_scale, dim=0).mean(dim=0)  # (B, T, num_cls, E)

        cls_outputs = self.out_ln(cls_outputs)
        return cls_outputs.flatten(-2)  # (B, T, num_cls * E)

    # ---- public API ---- #

    def _train_forward(self, embeddings: Tensor, d: Optional[Tensor] = None) -> Tensor:
        return self._aggregate_embeddings(embeddings, d=d)

    def _inference_forward(self, embeddings: Tensor, mgr_config: MgrConfig = None) -> Tensor:
        if mgr_config is None:
            mgr_config = MgrConfig()
        self.inference_mgr.configure(**mgr_config)
        representations = self.inference_mgr(
            self._aggregate_embeddings, inputs=OrderedDict([("embeddings", embeddings)])
        )
        return representations

    def forward(self, embeddings: Tensor, d: Optional[Tensor] = None, mgr_config: MgrConfig = None) -> Tensor:
        if self.training:
            return self._train_forward(embeddings, d)
        else:
            return self._inference_forward(embeddings, mgr_config)
