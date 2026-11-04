# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any, Generic, TypeVar, cast

import torch
from torch import Tensor

T = TypeVar("T", bound="Tensor | BufferList[Any]")


class BufferList(torch.nn.Module, Generic[T]):
    """Ordered list of tensors registered as PyTorch module state.

    Args:
        buffers: Tensors and buffer lists to register in order.
    """

    def __init__(
        self,
        buffers: Iterable[T] = (),
    ) -> None:
        super().__init__()
        for index, item in enumerate(buffers):
            if isinstance(item, BufferList):
                self.add_module(str(index), item)
            else:
                self.register_buffer(str(index), item)

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Any],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        layout = state_dict[f"{prefix}_extra_state"]
        materialized_keys = []
        for index, is_nested in enumerate(layout):
            if is_nested:
                continue
            name = str(index)
            key = f"{prefix}{index}"
            if name not in self._buffers:
                # Some tensor subclasses cannot use empty_like or copy_.
                self.register_buffer(name, state_dict.pop(key).clone())
                materialized_keys.append(key)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        for key in materialized_keys:
            if key in missing_keys:
                missing_keys.remove(key)

    def __getitem__(self, index: int) -> T:
        name = str(index)
        if name in self._modules:
            return cast(T, self.get_submodule(name))
        return cast(T, self.get_buffer(name))

    def __iter__(self) -> Iterator[T]:
        return (self[index] for index in range(len(self)))

    def __len__(self) -> int:
        return len(self._buffers) + len(self._modules)

    def get_extra_state(self) -> tuple[bool, ...]:
        r""":meta private:"""  # noqa: D415
        # Layout of the buffer list in a tuple of booleans.
        # True at positions that hold a nested BufferList rather than a tensor.
        return tuple(str(index) in self._modules for index in range(len(self)))

    def set_extra_state(self, state: object) -> None:
        r""":meta private:"""  # noqa: D415
        for index, is_nested in enumerate(cast(tuple[bool, ...], state)):
            name = str(index)
            if is_nested and name not in self._modules:
                self.add_module(name, BufferList())
