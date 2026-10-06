"""Pure-PyTorch mLSTM backend (stabilised chunkwise-parallel form).

Written for TabTune from the mLSTM equations (Beck et al., 2024, arXiv:2405.04517; chunkwise form as in
Beck et al., 2025, arXiv:2503.14376). It is not derived from ``mlstm_kernels``. Semantics match the
``chunkwise--native_autograd`` kernel used by TiRex-2 on CPU: zero initial state, stabiliser starting at 0,
queries scaled by ``head_dim ** -0.5`` and the normaliser ``max(|q . n|, exp(-m)) + eps``.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class mLSTMBackendConfig:
    chunk_size: int = 64
    eps: float = 1e-6


def mlstm_chunkwise(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    i: torch.Tensor,
    f: torch.Tensor,
    chunk_size: int = 64,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Causal mLSTM over a full sequence.

    ``q``, ``k``: ``(B, NH, S, DHQK)``; ``v``: ``(B, NH, S, DHV)``; ``i``, ``f``: input and forget gate
    pre-activations ``(B, NH, S)``. Returns the hidden states ``(B, NH, S, DHV)``.
    """
    B, NH, S, DHQK = q.shape
    DHV = v.shape[-1]
    q = q * DHQK**-0.5
    log_f = F.logsigmoid(f)

    # State after the previous chunk, stored scaled by exp(-m).
    c = q.new_zeros(B, NH, DHQK, DHV)
    n = q.new_zeros(B, NH, DHQK)
    m = q.new_zeros(B, NH)

    h_chunks = []
    for start in range(0, S, chunk_size):
        stop = min(start + chunk_size, S)
        q_c, k_c, v_c, i_c = q[:, :, start:stop], k[:, :, start:stop], v[:, :, start:stop], i[:, :, start:stop]
        # Log forget-gate decay from the start of the chunk up to and including each step.
        decay = log_f[:, :, start:stop].cumsum(-1)
        causal = torch.ones(stop - start, stop - start, dtype=torch.bool, device=q.device).tril()

        log_w = decay[..., :, None] - decay[..., None, :] + i_c[..., None, :]
        log_w = log_w.masked_fill(~causal, float("-inf"))
        log_state = decay + m[..., None]
        m_t = torch.maximum(log_state, log_w.amax(-1))

        w = torch.exp(log_w - m_t[..., None]) * (q_c @ k_c.transpose(-2, -1))
        state_scale = torch.exp(log_state - m_t)[..., None]
        numerator = state_scale * (q_c @ c) + w @ v_c
        denominator = state_scale * (q_c @ n[..., None]) + w.sum(-1, keepdim=True)
        h_chunks.append(numerator / (torch.maximum(denominator.abs(), torch.exp(-m_t)[..., None]) + eps))

        if stop < S:
            log_a = decay[..., -1:] - decay + i_c
            m_next = torch.maximum(decay[..., -1] + m, log_a.amax(-1))
            k_scaled = k_c * torch.exp(log_a - m_next[..., None])[..., None]
            carry = torch.exp(decay[..., -1] + m - m_next)
            c = carry[..., None, None] * c + k_scaled.transpose(-2, -1) @ v_c
            n = carry[..., None] * n + k_scaled.sum(-2)
            m = m_next

    return torch.cat(h_chunks, dim=2)


class mLSTMBackend(nn.Module):
    """Parameter-free module wrapping :func:`mlstm_chunkwise`."""

    config_class = mLSTMBackendConfig

    def __init__(self, config: mLSTMBackendConfig):
        super().__init__()
        self.config = config

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, i: torch.Tensor, f: torch.Tensor):
        return mlstm_chunkwise(q, k, v, i, f, chunk_size=self.config.chunk_size, eps=self.config.eps)

    def extra_repr(self) -> str:
        return f"{self.config}"
