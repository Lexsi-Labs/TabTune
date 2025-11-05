import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear(nn.Linear):
    def __init__(self, *args, replace_nan_by_zero=True, pad_to_dim=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.replace_nan_by_zero = replace_nan_by_zero
        self.pad_to_dim = pad_to_dim

    def forward(self, x, mask=None):
        if mask is not None:
            x = x.masked_fill(mask==0, 0.0)
        if self.replace_nan_by_zero:
            x = x.masked_fill(torch.isnan(x), 0.0)
        if self.pad_to_dim:
            x = F.pad(x, (0, self.in_features - x.shape[-1]), "constant", 0.0)
        return super().forward(x)


class FeedForward(nn.Module):
    def __init__(self, dim, in_dim=None, out_dim=None,
                 n_hid=1, activation="relu", bias=True,
                 replace_nan_by_zero=True, pad_to_dim=True):
        super().__init__()
        in_dim = dim if in_dim is None else in_dim
        out_dim = dim if out_dim is None else out_dim

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

        in_dims = [in_dim] + n_hid*[dim]
        out_dims = n_hid*[dim] + [out_dim]

        self._emb = Linear(in_dims[0], out_dims[0], bias=bias,
            replace_nan_by_zero=replace_nan_by_zero, pad_to_dim=pad_to_dim)

        _net = []
        for in_d, out_d in zip(in_dims[1:], out_dims[1:]):
            _net.extend([act(), nn.Linear(in_d, out_d, bias=bias)])
        self._net = nn.Sequential(*_net)

    def forward(self, x, mask=None):
        x = self._emb(x, mask=mask)
        return self._net(x)
