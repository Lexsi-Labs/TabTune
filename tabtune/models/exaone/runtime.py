"""Small runtime helpers shared by inference entry points."""

from collections.abc import Callable
from typing import Any

import contextlib

import torch


FULL_FEATURE_ATTENTION_ROWS = 2**31 - 1

# The FlashAttention kernel maps the independent batch axis onto a CUDA grid
# dimension the driver caps at 65535, so a longer batch is split across calls.
FLASH_ATTENTION_BATCH_LIMIT = 65_535


def run_in_chunks(
    operation: Callable[..., torch.Tensor],
    primary: torch.Tensor,
    *aligned: torch.Tensor,
    chunk_size: int,
    add_input: bool = False,
    axis: int = 0,
) -> torch.Tensor:
    """Apply ``operation`` to contiguous groups along ``axis`` and join results."""
    if not callable(operation):
        raise TypeError("operation must be callable")
    if not isinstance(primary, torch.Tensor):
        raise TypeError("primary must be a tensor")
    if isinstance(axis, bool) or not isinstance(axis, int):
        raise TypeError("axis must be an integer")
    if primary.ndim == 0:
        raise ValueError("primary must have a first axis")
    if not -primary.ndim <= axis < primary.ndim:
        raise ValueError("axis is out of range for primary")
    axis %= primary.ndim

    row_count = primary.shape[axis]
    for value in aligned:
        if not isinstance(value, torch.Tensor):
            raise TypeError("aligned inputs must be tensors")
        if value.ndim <= axis:
            raise ValueError("aligned inputs must span the chunk axis")
        if value.shape[axis] != row_count:
            raise ValueError("aligned inputs must have the same row count")

    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
        raise TypeError("chunk_size must be an integer")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not isinstance(add_input, bool):
        raise TypeError("add_input must be a bool")

    def checked_result(result: Any, source: torch.Tensor) -> torch.Tensor:
        if not isinstance(result, torch.Tensor):
            raise TypeError("operation must return a tensor")
        if result.ndim <= axis:
            raise ValueError("operation result must span the chunk axis")
        if result.shape[axis] != source.shape[axis]:
            raise ValueError("operation result has an unexpected row count")
        if add_input:
            if result.shape != source.shape:
                raise ValueError("residual result and input shapes must match")
            if result.dtype != source.dtype or result.device != source.device:
                raise RuntimeError("residual result and input types must match")
            result = result + source
        return result

    # TabTune modification: upstream wraps this whole helper in ``torch.no_grad()``
    # and detaches both return paths. It is the feature-attention row-chunking
    # path, engaged whenever ``feature_attention_row_chunk < rows`` -- which the
    # CUDA planner selects on its own under memory pressure. Under upstream's
    # unconditional ``no_grad`` a caller who enabled gradients would get
    # ``x.grad is None`` on some machines and a real gradient on others, decided
    # by how full the card happened to be. ``nullcontext`` under autograd keeps
    # the memory behaviour identical for inference (where grad is already off)
    # and makes the chunked path differentiable when a caller asked for it.
    with (contextlib.nullcontext() if torch.is_grad_enabled() else torch.no_grad()):
        if row_count <= chunk_size:
            result = checked_result(operation(primary, *aligned), primary)
            return result if torch.is_grad_enabled() else result.detach()

        # Blocks are copied into one preallocated destination rather than
        # collected and cat-ed: cat holds every block *and* its copy, flooring
        # the peak at two full lengths where a destination floors it at one.
        #
        # TabTune modification: that preallocate-and-``copy_`` strategy is an
        # in-place write into a tensor autograd did not create, repeated once per
        # chunk. Under grad it is at best fragile (version-counter bumps on a
        # tensor shared between iterations) and buys nothing, because the blocks
        # have to stay live for the backward pass either way. So the concatenating
        # form runs only when a caller enabled gradients; inference keeps
        # upstream's single-length peak exactly.
        grad_mode = torch.is_grad_enabled()
        destination: torch.Tensor | None = None
        blocks: list[torch.Tensor] = []
        expected_rest: tuple[int, ...] | None = None
        reference: torch.Tensor | None = None

        for start in range(0, row_count, chunk_size):
            length = min(chunk_size, row_count - start)
            source = torch.narrow(primary, axis, start, length)
            companions = tuple(
                torch.narrow(value, axis, start, length) for value in aligned
            )
            result = checked_result(operation(source, *companions), source)

            rest = tuple(result.shape[:axis]) + tuple(result.shape[axis + 1 :])
            if expected_rest is None:
                expected_rest = rest
                reference = result
                if not grad_mode:
                    shape = list(result.shape)
                    shape[axis] = row_count
                    destination = torch.empty(
                        shape, dtype=result.dtype, device=result.device
                    )
            elif (
                rest != expected_rest
                or result.dtype != reference.dtype
                or result.device != reference.device
            ):
                raise RuntimeError("operation results are incompatible")

            if grad_mode:
                blocks.append(result)
            else:
                assert destination is not None
                torch.narrow(destination, axis, start, length).copy_(result)
                del result  # so a finished block is not live during the next one

        if grad_mode:
            return torch.cat(blocks, dim=axis)
        assert destination is not None
        return destination.detach()
