# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
from collections.abc import Iterator
from typing import Literal

import torch


@contextlib.contextmanager
def inference_mode(
    mode: Literal["inference", "no_grad", "grad", "none"] = "inference",
) -> Iterator[None]:
    r"""Context manager to adjust PyTorch inference and autograd states.

    Args:
        mode: The desired mode. ``"inference"`` runs under
            :func:`torch.inference_mode` (falling back to :func:`torch.no_grad`
            inside compiled regions), ``"no_grad"`` runs under
            :func:`torch.no_grad`, ``"grad"`` runs under
            :func:`torch.enable_grad`, and ``"none"`` preserves the ambient
            PyTorch context.
    """
    if mode == "inference":
        # `torch.inference_mode` is not supported inside a compiled region:
        # https://github.com/pytorch/pytorch/issues/180823
        context = (
            torch.no_grad()
            if torch.compiler.is_compiling()
            else torch.inference_mode()
        )
    elif mode == "no_grad":
        context = torch.no_grad()
    elif mode == "grad":
        context = torch.enable_grad()
    else:
        assert mode == "none"
        context = contextlib.nullcontext()

    with context:
        yield
