import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1d(nn.Conv1d):
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
            x = F.pad(x, (0, self.kernel_size[0] - x.shape[-1] % self.kernel_size[0]), "constant", 0.0)
        return super().forward(x)
