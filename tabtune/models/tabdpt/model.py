from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import GELU, LayerNorm, Linear

from .utils import clip_outliers, flash_context, normalize_data


class TabDPTModel(nn.Module):
    def __init__(
        self,
        dropout: float,
        n_out: int,
        nhead: int,
        nhid: int,
        ninp: int,
        nlayers: int,
        num_features: int,
        use_flash: bool = True,
        clip_sigma: float = 4.
    ):
        super().__init__()
        self.n_out = n_out
        self.use_flash = use_flash
        self.ninp = ninp
        self.transformer_encoder = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    embed_dim=ninp,
                    num_heads=nhead,
                    ff_dim=nhid,
                )
                for _ in range(nlayers)
            ]
        )
        self.num_features = num_features
        self.encoder = nn.Linear(num_features, ninp)
        self.dropout = nn.Dropout(p=dropout)
        self.y_encoder = nn.Linear(1, ninp)
        self.head = nn.Sequential(nn.Linear(ninp, nhid), nn.GELU(), nn.Linear(nhid, n_out + 1))
        self.clip_sigma = clip_sigma

    @flash_context
    def forward(
        self,
        x_src: torch.Tensor,
        y_src: torch.Tensor,
        task: Literal["cls", "reg"],  # classification or regression
    ) -> torch.Tensor:
        x_src = x_src.transpose(0, 1)
        y_src = y_src.squeeze(-1).transpose(0, 1)
        eval_pos = y_src.shape[0]
        assert x_src.shape[1] == y_src.shape[1], "x_src and y_src must have the same batch size"
        x_src = clip_outliers(x_src, -1 if self.training else eval_pos, n_sigma=self.clip_sigma)
        x_src = normalize_data(x_src, -1 if self.training else eval_pos)
        x_src = clip_outliers(x_src, -1 if self.training else eval_pos, n_sigma=self.clip_sigma)
        if task == "reg":
            y_src, mean_y, std_y = normalize_data(y_src, return_mean_std=True)
            y_src = clip_outliers(y_src)

        x_src = torch.nan_to_num(x_src, nan=0)
        x_src = self.encoder(x_src)
        mean = (x_src**2).mean(dim=-1, keepdim=True)
        rms = torch.sqrt(mean)
        x_src = x_src / rms

        y_src = self.y_encoder(y_src.unsqueeze(-1))
        train_x = x_src[:eval_pos] + y_src
        src = torch.cat([train_x, x_src[eval_pos:]], 0)

        for layer in self.transformer_encoder:
            src = layer(src, eval_pos)
        pred = self.head(src)
        if task == "reg":
            pred = pred[eval_pos:, ..., -1]
        elif task == "cls":
            pred = pred[eval_pos:, ..., :-1]
        else:
            raise ValueError(f"Invalid task: {task}")

        if task == "reg":
            pred = pred * std_y + mean_y

        return pred

    @classmethod
    def load(cls, model_state, config, use_flash, clip_sigma: float = 4.):
        assert config.model.max_num_classes > 2
        model = TabDPTModel(
            dropout=config.training.dropout,
            n_out=config.model.max_num_classes,
            nhead=config.model.nhead,
            nhid=config.model.emsize * config.model.nhid_factor,
            ninp=config.model.emsize,
            nlayers=config.model.nlayers,
            num_features=config.model.max_num_features,
            use_flash=use_flash,
            clip_sigma=clip_sigma
        )

        model_state = {k.replace("_orig_mod.", ""): v for k, v in model_state.items()}
        model_state = {k.replace("model.", ""): v for k, v in model_state.items()}
        model.load_state_dict(model_state)
        model.to(config.env.device)
        model.eval()
        return model


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads
        self.kv_proj = nn.Linear(embed_dim, 2 * embed_dim, bias=False)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.attn_norm = LayerNorm(embed_dim)
        self.ff_norm = LayerNorm(embed_dim)
        self.ff = nn.Sequential(Linear(embed_dim, ff_dim), GELU(), Linear(ff_dim, embed_dim))
        self.q_norm = LayerNorm(self.head_dim)
        self.k_norm = LayerNorm(self.head_dim)

    def forward(self, x, eval_pos):
        x = x.transpose(0, 1)
        B, L, _ = x.size()
        h = self.attn_norm(x)
        q = self.q_proj(h)
        k, v = self.kv_proj(h[:, :eval_pos]).chunk(2, dim=-1)
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, eval_pos, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, eval_pos, self.num_heads, self.head_dim).transpose(1, 2)
        q, k = self.q_norm(q), self.k_norm(k)
        attn = F.scaled_dot_product_attention(q, k, v).transpose(1, 2)
        attn = self.out_proj(attn.reshape(B, L, self.num_heads * self.head_dim))
        x = x + attn
        x = x + self.ff(self.ff_norm(x))
        return x.transpose(0, 1)