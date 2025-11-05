from packaging import version
from typing import Optional
from collections import namedtuple
from inspect import signature

import numpy as np

from sklearn.metrics import roc_auc_score

import torch
from torch.nn.attention import SDPBackend

def set_device_config(flash):
    assert not (flash and version.parse(torch.__version__) < version.parse('2.0.0')), 'in order to use flash attention, you must be using pytorch 2.0 or above'

    # determine efficient attention configs for cuda and cpu
    cpu_config = [SDPBackend.FLASH_ATTENTION, SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]
    if torch.cuda.is_available() and flash:
        device_properties = torch.cuda.get_device_properties(torch.device('cuda'))

        if device_properties.major == 8 and device_properties.minor == 0:
            # A100 GPU detected, using flash attention if input tensor is on cuda
            cuda_config = [SDPBackend.FLASH_ATTENTION]
        else:
            # Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda
            cuda_config = [SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]
    else:
        cuda_config = []

    return cpu_config, cuda_config

def get_args(values):
    args = {}
    for i in getargspec(values['self'].__init__).args[1:]:
        args[i] = values[i]
    return args

def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand(other.size())
    return src

def scatter_sum(src: torch.Tensor,
                index: torch.Tensor,
                dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> torch.Tensor:
    index = broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index, src)
    else:
        return out.scatter_add_(dim, index, src)

def auc_metric(target, proba, multi_class='ovo'):
    if len(np.unique(target)) > 2:
        return roc_auc_score(target, proba, multi_class=multi_class)
    else:
        return roc_auc_score(target, proba[:, 1])
