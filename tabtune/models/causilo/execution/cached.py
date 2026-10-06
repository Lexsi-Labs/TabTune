"""Predict query rows against the fitted, reusable attention contexts."""

import math
from collections.abc import Callable, Iterator

import numpy as np
import torch

from ..data.dataset import PreparedDataset
from ..data.ensemble import EnsembleMember
from ..model import Model, ModelConfig
from .memory import FP32_BYTES, MAX_EXECUTION_ATTEMPTS, Stage, Workload, execution_budget
from .runner import ModelCache, ModelRunner


def query_memory(config: ModelConfig, features: int, training_rows: int, queries: int, device) -> int:
    """Estimate query workspace; resident K/V caches already reduce available memory."""
    groups = math.ceil(features / config.group_size)
    grid = queries * groups * config.width * FP32_BYTES
    encoding = queries * groups * config.group_size * (config.width + 6 * config.frequencies) * FP32_BYTES
    prediction = Workload(
        Stage.PREDICTION, queries, config.width * config.row_latents, training_rows
    ).bytes_per_item(device)
    return max(encoding + grid, 4 * grid + prediction)


def cached_predictions(
    model: Model,
    fitted: PreparedDataset,
    transformed: dict[str, np.ndarray],
    member_contexts: tuple[tuple[EnsembleMember, ModelCache], ...],
    device: torch.device,
    reduce_output: Callable[[torch.Tensor], torch.Tensor],
) -> Iterator[tuple[EnsembleMember, torch.Tensor]]:
    """Yield complete CPU outputs for explicit member/cache pairs supplied by FitState.

    Each cached member runs independently. Completed query chunks survive an
    OOM; only the current chunk is retried at a smaller size.
    """
    runner = ModelRunner(model)
    for member, context in member_contexts:
        queries = transformed[member.normalization][:, member.feature_order]
        pieces = []
        offset = 0
        budget = execution_budget(device)

        def cost(count):
            return query_memory(model.config, queries.shape[1], len(fitted.features), count, device)

        while offset < len(queries):
            low, high = 1, len(queries) - offset
            if cost(1) > budget:
                raise torch.cuda.OutOfMemoryError("Cached context leaves insufficient space for one query")
            while low < high:
                middle = (low + high + 1) // 2
                if cost(middle) <= budget:
                    low = middle
                else:
                    high = middle - 1
            count = low
            completed = False
            for _ in range(MAX_EXECUTION_ATTEMPTS):
                table = result = None
                try:
                    table = torch.as_tensor(
                        np.ascontiguousarray(queries[offset : offset + count]),
                        device=device,
                        dtype=torch.float32,
                    ).unsqueeze(0)
                    result = reduce_output(runner.predict_cached(table, context))[0].cpu()
                    pieces.append(result)
                    completed = True
                except torch.cuda.OutOfMemoryError:
                    # Drop failed device allocations but keep completed CPU pieces.
                    table = result = None
                if completed:
                    break
                if count == 1:
                    raise torch.cuda.OutOfMemoryError("Cached prediction cannot execute a single query")
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                budget = min(budget // 2, cost(count) - 1)
                count = max(1, count // 2)
            if not completed:
                raise torch.cuda.OutOfMemoryError("Cached prediction exhausted its memory recovery attempts")
            offset += count
        yield member, torch.cat(pieces, dim=0)
