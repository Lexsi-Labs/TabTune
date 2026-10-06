# Copyright (c) NXAI GmbH.
# This software may be used and distributed according to the terms of the NXAI Community License Agreement.

from dataclasses import fields

import torch


def round_up_to_next_multiple_of(x: int, multiple_of: int) -> int:
    return int(((x + multiple_of - 1) // multiple_of) * multiple_of)


def dataclass_from_dict(cls, dict: dict):
    class_fields = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in dict.items() if k in class_fields})


# Remove after Issue will be solved: https://github.com/pytorch/pytorch/issues/61474
def nanmax(tensor: torch.Tensor, dim: int | None = None, keepdim: bool = False) -> torch.Tensor:
    min_value = torch.finfo(tensor.dtype).min
    output = tensor.nan_to_num(min_value).max(dim=dim, keepdim=keepdim)
    return output.values


def nanmin(tensor: torch.Tensor, dim: int | None = None, keepdim: bool = False) -> torch.Tensor:
    max_value = torch.finfo(tensor.dtype).max
    output = tensor.nan_to_num(max_value).min(dim=dim, keepdim=keepdim)
    return output.values


def nanvar(tensor: torch.Tensor, dim: int | None = None, keepdim: bool = False) -> torch.Tensor:
    tensor_mean = tensor.nanmean(dim=dim, keepdim=True)
    output = (tensor - tensor_mean).square().nanmean(dim=dim, keepdim=keepdim)
    return output


def nanstd(tensor: torch.Tensor, dim: int | None = None, keepdim: bool = False) -> torch.Tensor:
    output = nanvar(tensor, dim=dim, keepdim=keepdim)
    output = output.sqrt()
    return output
