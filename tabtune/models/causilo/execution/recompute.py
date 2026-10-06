"""Bound feature memory by retaining column summaries and replaying row updates.

One member is evaluated at a time. Column passes read all training rows for a
few feature groups; row passes re-embed small row tiles across all groups.
Compact row-latent checkpoints and pooled row vectors stay on the execution
device. Host inputs are uploaded in tiles; only final predictions return to CPU.
No full (all rows, all groups, hidden width) grid is allocated on either device.
The temporary contexts belong to one prediction call, not the fitted state.
"""

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

import torch
from torch import Tensor

from ..nn.cache import BlockCache, KVPair
from .memory import (
    FP32_BYTES,
    MAX_EXECUTION_ATTEMPTS,
    RecomputeMemory,
    Stage,
    projection_workspace,
    tile_size,
)
from .precision import stage_autocast
from .runner import ModelRunner, inject_targets

logger = logging.getLogger(__name__)
Result = TypeVar("Result")


@dataclass
class RetryPlan:
    """Retry an operation at smaller sizes, retaining the stage's failure count.

    Each attempt must own its temporary tensors and leave shared inputs intact
    on failure. After the exception scope ends, its traceback and temporaries
    are released before the allocator cache is cleared. Callers commit completed
    tiles outside run(); a context builder instead retries its entire result.
    """

    limit: int
    device: torch.device
    error_message: str = "Minimum recomputation tile or retry limit reached"
    failures: int = 0

    def run(self, operation: Callable[[int], Result]) -> Result:
        while True:
            try:
                return operation(self.limit)
            except torch.cuda.OutOfMemoryError:
                self.failures += 1
            if self.limit == 1 or self.failures >= MAX_EXECUTION_ATTEMPTS:
                raise torch.cuda.OutOfMemoryError(self.error_message)
            if self.device.type == "cuda":
                with torch.cuda.device(self.device):
                    torch.cuda.empty_cache()
            self.limit = max(1, self.limit // 2)
            logger.info("Retrying recomputation with tile size %d", self.limit)


def _select_columns(state: BlockCache, section: slice) -> BlockCache:
    return BlockCache(tuple(KVPair(pair.key[:, section], pair.value[:, section]) for pair in state.layers))


class RecomputeRunner:
    """Evaluate host inputs with column summaries and compact row checkpoints."""

    def __init__(self, model, device: torch.device):
        self.model = model
        self.device = device
        self.runner = ModelRunner(model)
        self.memory = RecomputeMemory(model.config, device)

    def _limit(self, items: int, item_bytes: int, *, pending_bytes: int = 0, kernel_limited: bool = False) -> int:
        return tile_size(items, item_bytes, self.device, pending_bytes=pending_bytes, kernel_limited=kernel_limited)

    def _map(self, source: Tensor, operation, *, axis: int, limit: int, inplace: bool = False) -> Tensor:
        """Assemble row tiles on the execution device, retrying only the failed tile.

        Operations must not modify their input. The destination is committed
        only after the operation succeeds, including when it aliases source.
        Its allocation is retried with a smaller tile to reduce workspace, but
        the full destination must fit on the execution device.
        """
        items = source.shape[axis]
        destination = None
        retry = RetryPlan(limit, self.device)
        start = 0

        def compute_tile(size):
            length = min(size, items - start)
            part = source.narrow(axis, start, length).contiguous().to(self.device)
            result = operation(part, slice(start, start + length))
            output = destination
            if output is None:
                shape = list(result.shape)
                shape[axis] = items
                if (
                    inplace and tuple(shape) == source.shape
                    and result.dtype == source.dtype and result.device == source.device
                ):
                    output = source
                else:
                    output = result.new_empty(shape)
            return result, output, length

        while start < items:
            result, destination, length = retry.run(compute_tile)
            destination.narrow(axis, start, length).copy_(result)
            del result
            start += length
        return destination

    def _embed(self, table: Tensor, targets: Tensor, section: slice) -> Tensor:
        known = targets[:, section].to(self.device)
        return self.runner._embed(table, known if known.shape[1] else None)

    def _apply_columns(self, features: Tensor, component, state: BlockCache) -> Tensor:
        if self.model.config.task == "regression":
            features = features.float()
        with stage_autocast(self.model.config.task, Stage.COLUMN, self.device):
            return component.query(features.transpose(1, 2), state).transpose(1, 2)

    def _row_trace(self, features: Tensor) -> Tensor:
        with stage_autocast(self.model.config.task, Stage.ROW, self.device):
            return self.model.row.capture_latents(features)

    def _replay_row(self, features: Tensor, trace: Tensor | None) -> Tensor:
        with stage_autocast(self.model.config.task, Stage.ROW, self.device):
            return self.model.row.replay_features(features, trace)

    def _replay_training(self, features: Tensor, trace: Tensor | None) -> Tensor:
        """Update one group tile using bounded row batches, also respecting CUDA's grid limit."""
        limit = self._limit(features.shape[1], self.memory.row_workspace(features.shape[2]), kernel_limited=True)
        retry = RetryPlan(limit, self.device)
        start = 0

        def compute_tile(size):
            stop = min(start + size, features.shape[1])
            saved = trace[:, start:stop] if trace is not None else None
            return self._replay_row(features[:, start:stop], saved), stop

        while start < features.shape[1]:
            result, stop = retry.run(compute_tile)
            features[:, start:stop].copy_(result)
            del result
            start = stop
        return features

    def _summary_tile(self, table, targets, component, previous, trace):
        features = self._embed(table.to(self.device), targets, slice(0, targets.shape[1]))
        if previous is not None:
            features = self._apply_columns(features, self.model.columns[0], previous)
            features = self._replay_training(features, trace)
        if self.model.config.task == "regression":
            features = features.float()
        arranged = features.transpose(1, 2)
        del features
        summaries = []
        with stage_autocast(self.model.config.task, Stage.COLUMN, self.device):
            for index, layer in enumerate(component.layers):
                keys = layer.prepare(arranged)
                summaries.append(keys)
                # The final broadcast would be discarded; replay it in row passes.
                if index + 1 < len(component.layers):
                    arranged = layer.broadcast.query(arranged, keys)
        return BlockCache(tuple(summaries))

    def _column_summaries(self, table, targets, component, previous, trace):
        config = self.model.config
        count = targets.shape[1]
        groups = math.ceil(table.shape[-1] / config.group_size)
        summaries = self.memory.summary_bytes(groups, len(component.layers))
        # Individual summary pieces coexist with the concatenated result.
        limit = self._limit(
            groups, self.memory.column_workspace(count),
            pending_bytes=summaries + summaries, kernel_limited=True,
        )
        retry = RetryPlan(limit, self.device)
        pieces = []
        start = 0

        def compute_tile(size):
            stop = min(start + size, groups)
            selected = _select_columns(previous, slice(start, stop)) if previous is not None else None
            values = table[:, :count, start * config.group_size : stop * config.group_size]
            return self._summary_tile(values, targets, component, selected, trace), stop

        while start < groups:
            piece, stop = retry.run(compute_tile)
            pieces.append(piece)
            start = stop
        return BlockCache(tuple(
            KVPair(torch.cat([p.layers[i].key for p in pieces], dim=1),
                   torch.cat([p.layers[i].value for p in pieces], dim=1))
            for i in range(len(component.layers))
        ))

    def _encode(self, table: Tensor, targets: Tensor) -> Tensor:
        model, count = self.model, targets.shape[1]
        groups = math.ceil(table.shape[-1] / model.config.group_size)
        per_row = self.memory.row_workspace(groups)
        logger.info("Recompute encoder: first column summaries")
        first = self._column_summaries(table, targets, model.columns[0], None, None)

        def trace_rows(part, section):
            features = self._apply_columns(self._embed(part, targets, section), model.columns[0], first)
            return self._row_trace(features)

        logger.info("Recompute encoder: row latent checkpoints")
        trace = None
        if model.row.num_latent_states:
            limit = self._limit(count, per_row, pending_bytes=self.memory.trace_bytes(count), kernel_limited=True)
            trace = self._map(table[:, :count], trace_rows, axis=1, limit=limit)
        logger.info("Recompute encoder: second column summaries")
        second = self._column_summaries(table, targets, model.columns[1], first, trace)

        def pool_rows(part, section):
            features = self._apply_columns(self._embed(part, targets, section), model.columns[0], first)
            known = max(0, min(section.stop, count) - section.start)
            if known:
                saved = trace[:, section.start : section.start + known] if trace is not None else None
                training = self._replay_row(features[:, :known], saved)
                if known < features.shape[1]:
                    query = self.runner._row(model.row, features[:, known:], Stage.ROW)
                    features = torch.cat((training, query), dim=1)
                else:
                    features = training
            else:
                features = self.runner._row(model.row, features, Stage.ROW)
            features = self._apply_columns(features, model.columns[1], second)
            return self.runner._row(model.pool, features, Stage.POOL)

        logger.info("Recompute encoder: pooled row representations")
        limit = self._limit(
            table.shape[1], per_row, pending_bytes=self.memory.row_bytes(table.shape[1]), kernel_limited=True,
        )
        return self._map(table, pool_rows, axis=1, limit=limit)

    def _context(self, layer, training: Tensor) -> KVPair:
        """Project training rows in tiles while retaining the complete device K/V."""
        count, width = training.shape[1:]
        limit = self._limit(
            count, projection_workspace(1, width, self.model.config.prediction_heads),
            pending_bytes=2 * self.memory.row_bytes(count),
        )
        retry = RetryPlan(limit, self.device, error_message="Training K/V exceeds the streamed execution budget")

        def build_context(size):
            keys = None
            for start in range(0, count, size):
                stop = min(start + size, count)
                part = training[:, start:stop]
                prepared = layer.prepare(part)
                if keys is None:
                    shape = (*prepared.key.shape[:-2], count, prepared.key.shape[-1])
                    keys = KVPair(prepared.key.new_empty(shape), prepared.value.new_empty(shape))
                keys.key[..., start:stop, :].copy_(prepared.key)
                keys.value[..., start:stop, :].copy_(prepared.value)
                del part, prepared
            return keys

        return retry.run(build_context)

    def _predict_rows(self, rows: Tensor, targets: Tensor, reduce_output) -> Tensor:
        model, count = self.model, targets.shape[1]
        limit = self._limit(rows.shape[1], self.memory.prediction_workspace(count, final=False))

        def add_targets(part, section):
            known = targets[:, section].to(self.device)
            return inject_targets(part, model.row_target(known)) if known.shape[1] else part

        with stage_autocast(model.config.task, Stage.PREDICTION, self.device):
            rows = self._map(rows, add_targets, axis=1, limit=limit, inplace=True)
            for index, layer in enumerate(model.prediction.layers):
                logger.info("Recompute prediction: layer %d/%d", index + 1, len(model.prediction.layers))
                keys = self._context(layer, rows[:, :count])
                final = index + 1 == len(model.prediction.layers)
                queries = rows[:, count:] if final else rows
                pending = queries.shape[1] * model.config.outputs * FP32_BYTES if final else 0
                limit = self._limit(
                    queries.shape[1], self.memory.prediction_workspace(count, final=final), pending_bytes=pending,
                )

                def query(part, _):
                    result = layer.query(part, keys)
                    return reduce_output(model.head(result)) if final else result

                rows = self._map(queries, query, axis=1, limit=limit, inplace=not final)
                keys = queries = None
        return rows

    def predict(self, table: Tensor, targets: Tensor, reduce_output=lambda value: value) -> Tensor:
        """Return host predictions for one member without retaining a fitted K/V cache."""
        if table.device.type != "cpu" or targets.device.type != "cpu" or table.shape[0] != 1:
            raise ValueError("Recomputation requires one member's inputs on CPU")
        count = targets.shape[1]
        if count < 1 or table.shape[1] <= count:
            raise ValueError("Recomputation requires training and query rows")
        rows = self._encode(table, targets)
        return self._predict_rows(rows, targets, reduce_output).cpu()
