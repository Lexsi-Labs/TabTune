import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import FullAttention
from .feedforward import FeedForward
from .convolution import Conv1d


class TransformerBlock(nn.Module):
    """A Transformer block."""

    def __init__(self, d_model=512, n_heads=4, d_ff=2048,
                 dropout=0.1, activation="gelu",
                 norm_eps=1e-5, rope=False):
        """Initializes a new TransformerBlock instance.
        """
        super().__init__()
        self._attn = FullAttention(d_model, n_heads, dropout=dropout, rope=rope)
        self._ln_attn = nn.LayerNorm(d_model, eps=norm_eps)

        self._ff = FeedForward(d_ff, in_dim=d_model, out_dim=d_model,
            n_hid=1, activation=activation)
        self._ln_ff = nn.LayerNorm(d_model, eps=norm_eps)

    def forward(self, x, mask=None):
        x = x + self._attn(x, mask=mask)
        x = self._ln_attn(x)
        x = x + self._ff(x)
        x = self._ln_ff(x)

        return x


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size=128, d_model=512, n_heads=4, d_ff=2048,
                 dropout=0.0, activation="gelu", norm_eps=1e-5, rope=False):
        super().__init__()
        self.patch_size = patch_size

        if activation == "relu":
            act = nn.ReLU
        elif activation == "silu":
            act = nn.SiLU
        elif activation == "gelu":
            act = nn.GELU
        elif activation == "tanh":
            act = nn.Tanh
        elif activation == "sigmoid":
            act = nn.Sigmoid
        else:
            act = activation

        self._patchify = nn.Sequential(
            Conv1d(1, d_model, kernel_size=patch_size, stride=patch_size, padding=0),
            act(),
            nn.Conv1d(d_model, d_model, kernel_size=1, stride=1, padding=0)
        )

        self._emb = TransformerBlock(
            d_model=d_model, n_heads=n_heads, d_ff=d_ff,
            dropout=dropout, activation=activation,
            norm_eps=norm_eps, rope=rope
        )

        self._ln = nn.LayerNorm(d_model, eps=norm_eps)

    def forward(self, x):
        """
        x: (batch_size, data_size, feature_size)
        """
        _, l, f = x.size()
        x = self._patchify(x.view(-1, 1, f))

        _, d, p = x.size()
        x = x.view(-1, l, d, p).transpose(-1, -2)
        x = self._emb(x)

        x = self._ln(x.sum(-2))
        return x


class MixtureBlock(nn.Module):
    """A Mixture block."""

    def __init__(self, d_model=512, n_heads=4, d_ff=2048,
                 dropout=0.1, activation="gelu", temperature=0.2):
        """Initializes a new MixtureBlock instance.
        """
        super().__init__()
        self.n_heads = n_heads
        self.dropout = dropout
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5

        self._attn_logits = FeedForward(d_ff, in_dim=d_model, out_dim=d_model,
            n_hid=1, activation=activation)

        self._attn_gates = FeedForward(d_ff, in_dim=d_model, out_dim=d_model,
            n_hid=1, activation=activation)

        self.temperature = temperature

    def forward(self, hidden, split, quantile=0.3):
        """
        x: (batch_size, data_size, hidden_size)
        """
        b, l, _ = hidden.size()

        attn_gates = self._attn_gates(hidden)
        attn_gates = attn_gates.view(b, l, self.n_heads, -1).transpose(-3, -2)
        k_gates, q_gates = attn_gates[..., :split, :], attn_gates[..., split:, :]
        k_gates, q_gates = F.normalize(k_gates, dim=-1), F.normalize(q_gates, dim=-1)
        gates = torch.einsum(f"...ld, ...md -> ...lm", q_gates, k_gates)
        if self.training:
            gates = torch.distributions.RelaxedBernoulli(
                self.temperature, logits=gates
            ).rsample()
        else:
            gates = (
                gates >= torch.quantile(gates, quantile, dim=-1, keepdim=True)
            ).to(gates.dtype)

        attn_logits = self._attn_logits(hidden)
        attn_logits = attn_logits.view(b, l, self.n_heads, -1).transpose(-3, -2)
        k_logits, q_logits = attn_logits[..., :split, :], attn_logits[..., split:, :]
        logits = torch.einsum(f"...ld, ...md -> ...lm", q_logits, k_logits)
        probs = (logits * self.scale).softmax(dim=-1)
        probs = probs * gates
        probs = probs / probs.sum(-1, keepdim=True)

        probs = probs.mean(-3)
        return probs
