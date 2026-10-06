"""Partition independent leading axes without splitting attended sequences."""

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

from ..nn.cache import BlockCache, CacheResult, KVPair
from .memory import CUDA_MAX_BATCH_ITEMS, MAX_EXECUTION_ATTEMPTS, Stage, Workload, execution_budget


@dataclass
class ChunkPlan:
    """Mutable execution partition; ``limit`` shrinks after each allocation failure."""

    items: int
    limit: int
    budget: int
    item_bytes: int
    device: torch.device
    failures: int = 0
    destination_bytes: int = 0

    @classmethod
    def choose(cls, items: int, workload: Workload, device: torch.device):
        """Reserve space for the full destination whenever output chunks must be assembled."""
        budget = execution_budget(device)
        item_bytes = max(1, workload.bytes_per_item(device))
        kernel_limit = CUDA_MAX_BATCH_ITEMS if device.type == "cuda" else items
        limit = max(1, min(items, kernel_limit, budget // item_bytes))
        destination = workload.destination_bytes(items)
        if limit < items:
            limit = max(1, min(limit, (budget - destination) // item_bytes))
        return cls(items, limit, budget, item_bytes, device, destination_bytes=destination)

    def slices(self):
        """Visit every independent item once, preserving its original position."""
        for start in range(0, self.items, self.limit):
            yield slice(start, min(start + self.limit, self.items))

    def shrink(self):
        """Reduce the memory budget and choose a smaller partition after an allocation failure."""
        self.failures += 1
        if self.limit == 1 or self.failures >= MAX_EXECUTION_ATTEMPTS:
            raise torch.cuda.OutOfMemoryError("Minimum stage partition or retry limit reached")
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        self.budget = min(self.budget // 2, execution_budget(self.device))
        room = self.budget - self.destination_bytes
        self.limit = min(self.limit - 1, max(1, room // self.item_bytes))


class ChunkRunner:
    """Execute (*batch, sequence, width) tensors one group of batch items at a time."""

    def run(self, operation: Callable[[Tensor], Tensor], rows: Tensor, workload: Workload) -> Tensor:
        return self.run_indexed(lambda part, _: operation(part), rows, workload)

    def run_indexed(self, operation, rows: Tensor, workload: Workload) -> Tensor:
        """Call operation(chunk, slice) and restore the original leading axes.

        The slice indexes flattened batch items, allowing cache tensors to be
        sliced identically. Operations must be independent across those items;
        the final two input axes are kept intact. OOM retries discard the entire
        destination because a smaller partition restarts from item zero.
        """
        leading = rows.shape[:-2]
        items = math.prod(leading)
        packed = rows.reshape(items, *rows.shape[-2:])
        plan = ChunkPlan.choose(items, workload, rows.device)
        while True:
            destination = None
            result = None
            try:
                if plan.limit == items:
                    result = operation(packed, slice(0, items))
                    return result.reshape(*leading, *result.shape[1:])
                for section in plan.slices():
                    result = operation(packed[section], section)
                    if destination is None:
                        destination = result.new_empty((items, *result.shape[1:]))
                    destination[section] = result
                    del result
                return destination.reshape(*leading, *destination.shape[1:])
            except torch.cuda.OutOfMemoryError:
                # No failed output may stay live while the next plan is chosen.
                destination = result = None
            plan.shrink()


def reshape_keys(state: BlockCache, leading: tuple[int, ...]) -> BlockCache:
    """Reshape batch axes only; preserve (heads, context_length, head_width)."""
    return BlockCache(
        tuple(
            KVPair(
                layer.key.reshape(*leading, *layer.key.shape[-3:]),
                layer.value.reshape(*leading, *layer.value.shape[-3:]),
            )
            for layer in state.layers
        )
    )


class CacheRunner:
    """Build and query K/V caches with the same feature-group partitioning as rows."""

    def build_column(self, component, rows) -> CacheResult:
        """Return updated training rows and K/V tensors in their original batch layout."""
        leading = rows.shape[:-2]
        items = math.prod(leading)
        packed = rows.reshape(items, *rows.shape[-2:])
        retained_keys = sum(2 * layer.latents.numel() for layer in component.layers)
        workload = Workload(
            Stage.COLUMN, rows.shape[-2], rows.shape[-1], rows.shape[-2], cache_elements=retained_keys
        )
        plan = ChunkPlan.choose(items, workload, rows.device)
        while True:
            training = cache = result = None
            try:
                if plan.limit == items:
                    result = component.build_context(packed)
                    return CacheResult(
                        result.rows.reshape(*leading, *rows.shape[-2:]),
                        reshape_keys(result.cache, leading),
                    )
                for section in plan.slices():
                    result = component.build_context(packed[section])
                    if training is None:
                        training = result.rows.new_empty(packed.shape)
                        cache = BlockCache(
                            tuple(
                                KVPair(
                                    layer.key.new_empty((items, *layer.key.shape[1:])),
                                    layer.value.new_empty((items, *layer.value.shape[1:])),
                                )
                                for layer in result.cache.layers
                            )
                        )
                    training[section] = result.rows
                    # Cache slices and training slices must share the flattened
                    # item order, including the feature-group axis.
                    for destination, source in zip(cache.layers, result.cache.layers):
                        destination.key[section] = source.key
                        destination.value[section] = source.value
                    del destination, source
                    result = None
                return CacheResult(training.reshape(*leading, *rows.shape[-2:]), reshape_keys(cache, leading))
            except torch.cuda.OutOfMemoryError:
                training = cache = result = None
            plan.shrink()

    def query(self, component, rows, state: BlockCache, stage: Stage):
        """Select matching K/V items for each query chunk; retain the full context axis."""
        keys = reshape_keys(state, (-1,))
        workload = Workload(stage, rows.shape[-2], rows.shape[-1], keys.layers[0].key.shape[-2])

        def apply(part, section):
            selected = BlockCache(
                tuple(KVPair(layer.key[section], layer.value[section]) for layer in keys.layers)
            )
            return component.query(part, selected)

        return ChunkRunner().run_indexed(apply, rows, workload)
