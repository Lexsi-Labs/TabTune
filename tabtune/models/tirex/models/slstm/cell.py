# Copyright (c) NXAI GmbH.
# This software may be used and distributed according to the terms of the NXAI Community License Agreement.

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class sLSTMBlockConfig:
    embedding_dim: int
    num_heads: int
    ffn_proj_factor: float = 2.6667
    num_states: int = 4
    num_gates: int = 4

    @property
    def head_dim(self):
        return self.embedding_dim // self.num_heads


class sLSTMCell(nn.Module):
    def __init__(self, config: sLSTMBlockConfig):
        super().__init__()
        self.config = config

        self._recurrent_kernel_ = nn.Parameter(
            torch.empty((config.num_heads, config.head_dim, config.num_gates * config.head_dim), dtype=None)
        )

        self._bias_ = nn.Parameter(torch.empty((config.num_heads * config.num_gates * config.head_dim), dtype=None))

        self._impl_forward_torch = sLSTMCellTorch.slstm_forward

    def forward(self, input: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        input = self._get_input(input)
        state = self._get_state(input, state)

        output, state = self._impl_torch(input, state)

        return self._permute_output(output).to(input.dtype), state.to(input.dtype)

    def _impl_torch(self, input: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        input = input.to(dtype=torch.bfloat16)
        state = state.to(dtype=torch.bfloat16)
        recurrent_kernel = self._recurrent_kernel_.to(dtype=torch.bfloat16)
        bias = self._bias_.to(dtype=torch.float32)

        input = input.view(input.shape[0], input.shape[1], -1)
        bias = (
            bias.reshape(self.config.num_heads, self.config.num_gates, self.config.head_dim)
            .permute(1, 0, 2)
            .reshape(-1)
        )

        return self._impl_forward_torch(input, state, recurrent_kernel, bias)

    def _get_input(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.config.embedding_dim * self.config.num_gates, (
            f"Input size mismatch: Expected input size {self.config.embedding_dim * self.config.num_gates}, but got {x.size(-1)}."
        )
        return x.view(x.shape[0], x.shape[1], self.config.num_gates, self.config.num_heads, -1).permute(1, 0, 2, 3, 4)

    def _get_state(self, input: torch.Tensor, state: torch.Tensor | None) -> torch.Tensor:
        B = input.shape[1]
        if state is None:
            state = torch.zeros(
                (self.config.num_states, B, self.config.embedding_dim),
                dtype=input.dtype,
                device=input.device,
            )

        assert state.shape == (self.config.num_states, B, self.config.embedding_dim)
        return state

    def _permute_output(self, output: torch.Tensor) -> torch.Tensor:
        output = output.view(output.shape[0], output.shape[1], self.config.num_heads, self.config.head_dim)
        return output.permute(1, 2, 0, 3)


class sLSTMCellTorch:
    @staticmethod
    def slstm_forward(
        x: torch.Tensor,  # [S, B, G*I]
        states: torch.Tensor,  # [4, B, H] only the first is used for recurrence!
        R: torch.Tensor,  # [K, R*H, H] - K num_heads
        b: torch.Tensor,  # [T*H]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_gates = 4
        num_heads = R.shape[0]
        S, B, _ = x.shape
        H = R.shape[1] * num_heads
        assert states.shape == (num_gates, B, H)

        states = states.to(R.dtype).unbind(dim=0)
        output = []
        for i in range(S):
            Ry = (
                torch.einsum("bhd,hdn->bhn", states[0].view(B, num_heads, -1), R)
                .view(B, num_heads, num_gates, -1)
                .transpose(1, 2)
                .reshape(B, -1)
            )
            states = sLSTMCellTorch.slstm_forward_pointwise(
                x[i].float(), Ry.float(), b.float(), [s.float() for s in states]
            )
            states = [s.to(dtype=R.dtype) for s in states]
            output.append(states[0])

        return torch.stack(output), torch.stack(states)  # (S, B, H), 4 x (B, H)

    @staticmethod
    def slstm_forward_pointwise(
        Wx: torch.Tensor,  # dim [B, 4*H]
        Ry: torch.Tensor,  # dim [B, 4*H]
        b: torch.Tensor,  # dim [1, 4*H]
        states: torch.Tensor,  # dim 4 x [B, H]
    ) -> list[torch.Tensor]:
        y, c, n, m = states

        raw = Wx + Ry + b
        iraw, fraw, zraw, oraw = torch.unbind(raw.view(raw.shape[0], 4, -1), dim=1)

        # Equations reference the xlstm paper on page 4: https://arxiv.org/pdf/2405.04517
        logfplusm = m + F.logsigmoid(torch.clamp(fraw, max=15))  # eq 15 # Clamp to avoid subnomals
        mnew = torch.where(torch.all(n == 0.0), iraw, torch.max(iraw, logfplusm))  # eq 15
        ogate = torch.sigmoid(oraw)  # eq 14
        igate = torch.minimum(torch.exp(iraw - mnew), torch.ones_like(iraw))  # eq 16
        fgate = torch.minimum(torch.exp(logfplusm - mnew), torch.ones_like(iraw))  # eq 17
        zgate = torch.tanh(zraw)  # eq 11
        cnew = fgate * c + igate * zgate  # eq 8
        nnew = fgate * n + igate  # eq 9
        hnew = ogate * cnew / nnew  # eq 10

        return [hnew, cnew, nnew, mnew]  # 4 x (B, H)
