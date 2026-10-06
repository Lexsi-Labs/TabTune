"""Conservative execution budgets; sizing changes batching, never model settings."""

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import psutil
import torch

if TYPE_CHECKING:
    from ..model import ModelConfig

FP32_BYTES = 4
EXECUTION_MEMORY_FRACTION = 0.8
MAX_EXECUTION_ATTEMPTS = 7
# Bound the flattened attention batch to fit the CUDA kernel grid dimension.
CUDA_MAX_BATCH_ITEMS = 65535

# Fixed heuristic workspace allowances, not dimensions read from ModelConfig.
# They preserve the existing sizing policy; actual allocation failures are
# handled by reducing partitions in the execution runners.
COLUMN_CONTEXT_ELEMENTS_PER_WIDTH = 128 * 8
PREDICTION_CONTEXT_BYTES_PER_ROW = 512 * 8


class Stage(Enum):
    """Stages with different activation lifetimes and workspace estimates."""

    COLUMN = "column"
    ROW = "row"
    POOL = "pool"
    PREDICTION = "prediction"


def embedding_workspace(config: "ModelConfig") -> int:
    """Bytes per row/group for sum-before-projection embedding, bounded in FP32.

    Count phase, sine/cosine, concatenation/masking, projection and missing
    buffers by shape. FP32 also covers input conversion under mixed precision.
    """
    values = config.group_size * FP32_BYTES
    missing = config.group_size  # Boolean mask.
    phase = config.group_size * config.frequencies * FP32_BYTES
    waves = 2 * phase  # Sine and cosine channels.
    summed = 2 * config.frequencies * FP32_BYTES
    signal = config.width * FP32_BYTES
    return values + missing + max(
        phase + waves + waves,  # Concatenated waves and their masked copy.
        waves + summed + signal,
        signal + values + signal,  # Signal, converted mask and missing embedding.
    )


def projection_workspace(rows: int, width: int, heads: int) -> int:
    """Normalize inputs, project K/V and normalize keys, bounded in FP32."""
    normalized = rows * width * FP32_BYTES
    packed_kv = 2 * normalized
    precise_key = squared_key = normalized_key = normalized
    head_statistics = rows * heads * FP32_BYTES
    return normalized + packed_kv + precise_key + squared_key + normalized_key + head_statistics


def attention_workspace(
    queries: int, context: int, width: int, heads: int, expansion: int, device: torch.device,
    *, project_context: bool = False,
) -> int:
    """Temporary attention/FFN tensors; callers account for resident inputs/K/V.

    FP32 bounds residuals, head normalization and autocast intermediates. The
    backend assumption matches Workload: allow dense scores on CPU/older CUDA;
    modern CUDA uses a streaming attention estimate, with OOM retry for backend
    workspace or dispatch differences. This estimates memory, not kernel choice.
    """
    vector = queries * width * FP32_BYTES
    expanded = queries * width * expansion * FP32_BYTES
    # Normalized query, projected/scaled query, attended/merged values and output.
    normalized_query = projected_query = scaled_query = attended = merged = update = vector
    attention = normalized_query + projected_query + scaled_query + attended + merged + update
    # Residual, normalized input, gate/SiLU/value/product and projected result.
    gate = activated_gate = value = product = expanded
    feedforward = vector + vector + gate + activated_gate + value + product + vector
    statistics = queries * heads * FP32_BYTES
    streaming = False
    if device.type == "cuda":
        with torch.cuda.device(device):
            streaming = torch.cuda.is_bf16_supported(including_emulation=False)
    scores = 0 if streaming else heads * queries * context * FP32_BYTES
    # Score and softmax buffers may coexist in the dense backend.
    attention += scores + scores + statistics
    if project_context:
        attention += projection_workspace(context, width, heads)
    return max(attention, feedforward)


@dataclass(frozen=True)
class RecomputeMemory:
    """Shape-based stage workspaces and pending buffers for one ensemble member.

    Retained buffers use FP32 bounds; live tensors are already reflected in
    execution_budget and must not be subtracted again. The existing execution
    memory fraction leaves headroom for backend workspaces and allocator costs.
    """

    config: "ModelConfig"
    device: torch.device

    def trace_bytes(self, rows: int) -> int:
        c = self.config
        return rows * (c.row_depths[0] - 1) * c.row_latents * c.width * FP32_BYTES

    def row_bytes(self, rows: int) -> int:
        return rows * self.config.row_latents * self.config.width * FP32_BYTES

    def summary_bytes(self, groups: int, depth: int) -> int:
        c = self.config
        return groups * depth * 2 * c.column_latents * c.width * FP32_BYTES

    def column_workspace(self, rows: int) -> int:
        c = self.config
        features = rows * c.width * FP32_BYTES
        gather = attention_workspace(
            c.column_latents, rows, c.width, c.column_heads, c.expansion, self.device, project_context=True,
        )
        broadcast = attention_workspace(rows, c.column_latents, c.width, c.column_heads, c.expansion, self.device)
        return features + max(rows * embedding_workspace(c), gather, broadcast)

    def row_workspace(self, groups: int) -> int:
        c = self.config
        features = groups * c.width * FP32_BYTES
        row_update = attention_workspace(
            groups + c.row_latents, c.row_latents, c.width, c.row_heads, c.expansion, self.device,
            project_context=True,
        )
        latent_update = attention_workspace(
            c.row_latents, groups + c.row_latents, c.width, c.row_heads, c.expansion, self.device,
            project_context=True,
        )
        broadcast = groups * attention_workspace(
            1, c.column_latents, c.width, c.column_heads, c.expansion, self.device,
        )
        # Row capture retains each latent state and stacks a second copy.
        checkpoints = self.trace_bytes(1)
        # Keep the input and updated feature tiles while replaying/splitting rows.
        return features + features + checkpoints + checkpoints + max(
            groups * embedding_workspace(c), broadcast, row_update, latent_update,
        )

    def prediction_workspace(self, training: int, *, final: bool) -> int:
        c = self.config
        width = c.width * c.row_latents
        attention = attention_workspace(1, training, width, c.prediction_heads, c.expansion, self.device)
        # The head uses normalized rows, a width*2 hidden/GELU pair and logits.
        normalized = width * FP32_BYTES
        hidden = 2 * normalized
        logits = c.outputs * FP32_BYTES
        head = normalized + hidden + hidden + logits if final else 0
        # Classification target embedding materializes int64 and FP32 one-hot labels.
        targets = c.outputs * (torch.int64.itemsize + FP32_BYTES) if c.task == "classification" else FP32_BYTES
        return max(attention, head, targets + normalized + normalized)


def tile_size(
    items: int, item_bytes: int, device: torch.device, *, pending_bytes: int = 0, kernel_limited: bool = False,
) -> int:
    """Reserve unallocated full buffers before sizing temporary work per item."""
    room = execution_budget(device) - pending_bytes
    if room <= 0:
        raise torch.cuda.OutOfMemoryError("Retained recomputation states exceed the execution budget")
    kernel_limit = CUDA_MAX_BATCH_ITEMS if kernel_limited and device.type == "cuda" else items
    return max(1, min(items, kernel_limit, room // max(1, item_bytes)))


@dataclass(frozen=True)
class Workload:
    """Memory estimate for one independently executable item.

    ``rows`` is the attended sequence length: table rows for column/prediction
    stages, feature groups for row/pool stages. Leading batch axes are counted
    separately as ``items`` by ChunkPlan. Byte counts conservatively assume FP32
    intermediates even when attention uses lower precision.
    """

    stage: Stage
    rows: int
    width: int
    context_rows: int = 0
    output_elements: int | None = None
    cache_elements: int = 0

    def destination_bytes(self, items: int) -> int:
        """Space for the assembled output and any K/V tensors kept across chunks."""
        elements = self.rows * self.width if self.output_elements is None else self.output_elements
        return items * (elements + self.cache_elements) * FP32_BYTES

    def bytes_per_item(self, device: torch.device) -> int:
        """Estimate temporary activations plus attention workspace for one item."""
        activation_copies = {Stage.COLUMN: 12, Stage.ROW: 16, Stage.POOL: 7, Stage.PREDICTION: 15}[self.stage]
        payload = self.rows * self.width * FP32_BYTES * activation_copies
        if self.stage == Stage.COLUMN:
            payload += COLUMN_CONTEXT_ELEMENTS_PER_WIDTH * self.width * FP32_BYTES
        if self.stage == Stage.PREDICTION:
            payload += self.context_rows * PREDICTION_CONTEXT_BYTES_PER_ROW
            native_bfloat = False
            if device.type == "cuda":
                with torch.cuda.device(device):
                    native_bfloat = torch.cuda.is_bf16_supported(including_emulation=False)
            if not native_bfloat:
                # Allow for materialized query-by-context attention matrices.
                payload += 4 * self.rows * self.context_rows * FP32_BYTES
        return payload


def available_bytes(device: torch.device) -> int:
    """Count reusable allocator memory, bounded by both physical and process limits."""
    if device.type == "cpu":
        return psutil.virtual_memory().available
    free, total = torch.cuda.mem_get_info(device)
    allocated = torch.cuda.memory_allocated(device)
    reusable = torch.cuda.memory_reserved(device) - allocated
    allocator_room = int(total * torch.cuda.get_per_process_memory_fraction(device)) - allocated
    return max(0, min(free + reusable, allocator_room))


def execution_budget(device: torch.device) -> int:
    """Leave headroom for allocations outside the stage workspace estimate."""
    return int(available_bytes(device) * EXECUTION_MEMORY_FRACTION)
