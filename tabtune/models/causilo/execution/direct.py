"""Predict with training context recomputed for each ensemble/query batch."""

import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

from ..data.dataset import PreparedDataset
from ..data.ensemble import EnsembleMember
from ..model import Model, ModelConfig
from .memory import FP32_BYTES, MAX_EXECUTION_ATTEMPTS, Stage, Workload, embedding_workspace, execution_budget
from .recompute import RecomputeRunner
from .runner import ModelRunner


@dataclass(frozen=True)
class BatchPlan:
    """Execution route and the member/query batch sizes it should use."""

    members: int
    queries: int
    mode: Literal["direct", "recompute"] = "direct"


@dataclass(frozen=True)
class BatchPlanner:
    """Estimate peak live tensors and favor whole-query ensemble batches."""

    config: ModelConfig
    training_rows: int
    features: int

    def estimate(self, members: int, queries: int, device: torch.device) -> int:
        """Estimate the peak stage, including resident feature grids shared by chunks."""
        config = self.config
        rows = self.training_rows + queries
        groups = math.ceil(self.features / config.group_size)
        grid = rows * groups * config.width * FP32_BYTES
        encoding = rows * groups * embedding_workspace(config)
        context = Workload(Stage.COLUMN, rows, config.width, self.training_rows).bytes_per_item(device)
        prediction = Workload(
            Stage.PREDICTION, rows, config.width * config.row_latents, self.training_rows
        ).bytes_per_item(device)
        # Stage partitions share the resident feature grid and one destination grid.
        return members * max(encoding + grid, 3 * grid + context, prediction + grid)

    def choose(self, members: int, queries: int, budget: int, device: torch.device) -> BatchPlan:
        """Reduce members, then compare query batching with encoder replay work."""
        for count in range(members, 0, -1):
            if self.estimate(count, queries, device) <= budget:
                return BatchPlan(count, queries)
        replay = BatchPlan(1, queries, mode="recompute")
        if self.estimate(1, 1, device) > budget:
            return replay
        low, high = 1, queries
        while low < high:
            middle = (low + high + 1) // 2
            if self.estimate(1, middle, device) <= budget:
                low = middle
            else:
                high = middle - 1
        batches = math.ceil(queries / low)
        training, extra = self._training_work()
        if self._recompute_state_bytes(queries) < budget and (batches - 1) * training > extra:
            return replay
        return BatchPlan(1, low)

    def _training_work(self) -> tuple[int, int]:
        """Estimate direct training MACs and the extra MACs needed for replay.

        Count dominant attention/linear products, not elapsed time. Query work
        is common to both routes. Kernel efficiency, transfers, normalization
        and elementwise work are omitted; ties favor direct query batching.
        """
        config, count = self.config, self.training_rows
        width, latents = config.width, config.row_latents
        groups = math.ceil(self.features / config.group_size)
        rounds = config.row_depths[0] - 1
        first, second = config.column_depths

        def attention(queries, context, *, feedforward=False):
            work = 2 * (queries + context) * width**2 + 2 * queries * context * width
            return work + (3 * config.expansion * queries * width**2 if feedforward else 0)

        embedding = groups * 2 * config.frequencies * width
        broadcast = (2 + 3 * config.expansion) * width**2 + 2 * config.column_latents * width
        gather = attention(config.column_latents, count, feedforward=True) + 2 * config.column_latents * width**2
        finish = attention(groups, latents)
        update = attention(latents, groups + latents, feedforward=True)
        row = rounds * (attention(groups + latents, latents) + update) + finish
        pool = config.row_depths[1] * update
        encoder = count * (embedding + row + pool) + groups * (first + second) * (gather + count * broadcast)

        # Every layer prepares training K/V; only nonfinal layers update the
        # training rows. This attention term grows quadratically with count.
        prediction_width = width * latents
        prediction = 2 * config.prediction_depth * count * prediction_width**2
        prediction += max(0, config.prediction_depth - 1) * count * (
            (2 + 3 * config.expansion) * prediction_width**2 + 2 * count * prediction_width
        )

        # Recompute embeds training 4 times (3 without a latent trace). Column
        # summaries skip their final broadcast; row features are replayed twice.
        extra_embeddings = 2 + int(rounds > 0)
        extra_broadcasts = (extra_embeddings - 1) * first + max(0, first - 1) + max(0, second - 1)
        extra = count * (
            extra_embeddings * embedding + groups * extra_broadcasts * broadcast
            + (2 * (rounds + 1) - 1) * finish
        )
        return encoder + prediction, extra

    def _recompute_state_bytes(self, queries: int) -> int:
        """Conservatively budget retained states before preferring recomputation.

        Include overlapping row buffers and unreduced outputs in FP32. Stage
        workspaces are sized separately by RecomputeRunner and can still OOM.
        """
        config, count = self.config, self.training_rows
        width = config.width * config.row_latents
        groups = math.ceil(self.features / config.group_size)
        trace = count * (config.row_depths[0] - 1) * width
        rows = 2 * (count + queries) * width
        keys = 2 * count * width
        summaries = 2 * groups * sum(config.column_depths) * config.column_latents * config.width
        return (trace + rows + keys + summaries + queries * config.outputs) * FP32_BYTES


def _feature_batch(
    fitted: PreparedDataset,
    transformed: dict[str, np.ndarray],
    members: tuple[EnsembleMember, ...],
    query_slice: slice,
) -> np.ndarray:
    """Pack (members, training + query rows, features) with training rows first."""
    tables = []
    for member in members:
        training = fitted.training_table(member.normalization)
        queries = transformed[member.normalization][query_slice]
        table = np.concatenate((training, queries))
        tables.append(table[:, member.feature_order])
        # Only the permuted copies need to stay live until the batch is stacked.
        del table, training, queries
    return np.stack(tables)


def direct_predictions(
    model: Model,
    fitted: PreparedDataset,
    transformed: dict[str, np.ndarray],
    device: torch.device,
    reduce_output: Callable[[torch.Tensor], torch.Tensor],
) -> Iterator[tuple[EnsembleMember, torch.Tensor]]:
    """Yield each member with all its query outputs on CPU, in execution order.

    An OOM restarts the current member batch, discarding its partial query
    outputs. Previously yielded member batches remain complete and valid.
    """
    members = fitted.members_by_normalization()
    query_count = len(next(iter(transformed.values())))
    footprint = BatchPlanner(model.config, len(fitted.features), fitted.features.shape[1])
    budget = execution_budget(device)
    runner = ModelRunner(model)
    start = 0
    while start < len(members):
        completed = False
        for _ in range(MAX_EXECUTION_ATTEMPTS):
            plan = footprint.choose(len(members) - start, query_count, budget, device)
            if plan.mode == "recompute":
                recompute = RecomputeRunner(model, device)
                for member in members[start:]:
                    tables = _feature_batch(fitted, transformed, (member,), slice(0, query_count))
                    targets = np.stack([fitted.targets_for(member)])
                    prediction = recompute.predict(
                        torch.from_numpy(tables).float(), torch.from_numpy(targets).float(), reduce_output
                    )
                    del tables, targets
                    yield member, prediction[0]
                return
            selected = members[start : start + plan.members]
            pieces = []
            tensor = result = target_tensor = None
            try:
                targets = np.stack([fitted.targets_for(member) for member in selected])
                target_tensor = torch.as_tensor(targets, device=device, dtype=torch.float32)
                for offset in range(0, query_count, plan.queries):
                    tables = _feature_batch(
                        fitted, transformed, selected, slice(offset, offset + plan.queries)
                    )
                    tensor = torch.as_tensor(tables, device=device, dtype=torch.float32)
                    result = runner.predict(tensor, target_tensor)
                    pieces.append(reduce_output(result).cpu())
                    del tensor, result
                completed = True
            except torch.cuda.OutOfMemoryError:
                # Release device references before empty_cache and replanning.
                # Clear this batch's outputs so retried rows cannot be duplicated.
                pieces.clear()
                tensor = result = target_tensor = None
            if completed:
                break
            torch.cuda.empty_cache()
            budget = min(budget // 2, footprint.estimate(plan.members, plan.queries, device) - 1)
        if not completed:
            raise torch.cuda.OutOfMemoryError("Prediction exhausted its memory recovery attempts")
        result = torch.cat(pieces, dim=1)
        for member, prediction in zip(selected, result):
            yield member, prediction
        start += plan.members
