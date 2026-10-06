# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 NXAI GmbH
# Korbinian Poeppel

"""sLSTM recurrence ported from the "vanilla" backend of flashrnn 1.0.8 (https://github.com/NX-AI/flashrnn).

Sources: ``flashrnn/flashrnn/vanilla/{__init__,slstm}.py`` and the shape handling of
``flashrnn/flashrnn/flashrnn.py``. See ATTRIBUTION.md.
"""

import torch


def slstm_pointwise(
    Wx: torch.Tensor,  # [B, G, H, D]
    Ry: torch.Tensor,  # [B, G, H, D]
    b: torch.Tensor,  # [G, H, D]
    states: torch.Tensor,  # [B, S, H, D]
) -> torch.Tensor:
    raw = Wx + Ry + b
    y, c, n, m = torch.unbind(states, dim=1)
    iraw, fraw, zraw, oraw = torch.unbind(raw, dim=1)
    logfplusm = torch.nn.functional.logsigmoid(fraw) + m

    mnew = torch.where(torch.all(n == 0.0), iraw, torch.max(iraw, logfplusm))
    ogate = torch.sigmoid(oraw)
    igate = torch.exp(iraw - mnew)
    fgate = torch.exp(logfplusm - mnew)
    cnew = fgate * c + igate * torch.tanh(zraw)
    nnew = torch.maximum(fgate * n + igate, torch.ones_like(n))
    ynew = ogate * cnew / nnew

    return torch.stack((ynew, cnew, nnew, mnew), dim=1)


def slstm_forward(
    Wx: torch.Tensor,  # [B, T, G, H, D] gate pre-activations from the input, gate order (i, f, z, o)
    R: torch.Tensor,  # [H, P, G, D] recurrent kernel, P = D
    b: torch.Tensor,  # [H, G, D]
) -> torch.Tensor:  # [B, T, H, D] hidden states y
    """Run the sLSTM from a zero initial state and return the hidden state of every step."""
    batch_dim, _, num_gates, num_heads, head_dim = Wx.shape
    num_states = 4
    states = Wx.new_zeros(batch_dim, num_states, num_heads, head_dim)
    R = R.reshape(num_heads, head_dim, num_gates * head_dim)
    b = b.permute(1, 0, 2)

    ys = []
    for Wx_t in Wx.unbind(dim=1):
        Ry = states[:, 0].transpose(0, 1).bmm(R).view(num_heads, batch_dim, num_gates, head_dim).permute(1, 2, 0, 3)
        states = slstm_pointwise(Wx_t, Ry, b, states)
        ys.append(states[:, 0])
    return torch.stack(ys, dim=1)
