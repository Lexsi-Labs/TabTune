"""Shape-derived CUDA inference memory estimates.

The estimator mirrors the inference-time tensor lifetimes in the encoder,
transformer attention, feed-forward, and support-cache paths.  It deliberately
contains no device-model or installed-VRAM calibration: hardware capacity is a
budget supplied by the caller, not an input to the tensor-size calculation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

import torch

from ..config import ModelConfig
from ..runtime import FULL_FEATURE_ATTENTION_ROWS


_CUDA_LARGE_BLOCK_BYTES = 2 * 1024**2


class PeakMemoryMode(str, Enum):
    """One model-call phase whose CUDA peak is estimated independently."""

    JOINED = "joined"
    CACHE_BUILD_GPU = "cache_build_gpu"
    CACHE_BUILD_CPU = "cache_build_cpu"
    CACHED_PADDED_GPU = "cached_padded_gpu"
    CACHED_PADDED_CPU = "cached_padded_cpu"
    CACHED_COMPACT_GPU = "cached_compact_gpu"
    CACHED_COMPACT_CPU = "cached_compact_cpu"
    SUPPORT_ONLY_BUILD_GPU = "support_only_build_gpu"


@dataclass(frozen=True, slots=True)
class PeakMemoryEstimate:
    """Estimated tensor peak and its allocator-level budget projection."""

    allocated_bytes: int
    projected_reserved_bytes: int
    target_vram_bytes: int
    limiting_phase: str
    fits: bool


def _positive_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _nonnegative_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _element_size(dtype: torch.dtype) -> int:
    if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
        raise TypeError("dtype must be a floating torch dtype")
    return torch.empty((), dtype=dtype).element_size()


def _round_up(value: int, quantum: int) -> int:
    return ((value + quantum - 1) // quantum) * quantum


def _ffn_peak(
    *,
    input_bytes: int,
    token_count: int,
    width: int,
    expanded_width: int,
    element_size: int,
    feedforward_token_chunk: int,
) -> int:
    """Live bytes added while ``CrossAxisSummaryLayer._feedforward`` runs."""
    chunk_tokens = min(token_count, feedforward_token_chunk)
    chunk_state = chunk_tokens * width * element_size
    chunk_expanded = chunk_tokens * expanded_width * element_size
    # The input remains live. Completed output pieces are retained until their
    # final cat, while GELU briefly overlaps its input and output.
    compute_peak = 2 * input_bytes + 2 * chunk_expanded + chunk_state
    concatenate_peak = 3 * input_bytes
    return max(compute_peak, concatenate_peak)


def _working_set(
    config: ModelConfig,
    *,
    support_rows: int,
    query_rows: int,
    feature_count: int,
    members: int,
    element_size: int,
    transformer_rows: int,
    compact_transformer: bool,
    persistent_gpu_cache: bool,
    cpu_cache_transfer: bool,
    feedforward_token_chunk: int,
    feature_attention_row_chunk: int,
    output_width: int,
    decoder_hidden_width: int,
) -> dict[str, int]:
    groups = math.ceil(feature_count / config.columns_per_group)
    state_groups = groups + 1
    joined_rows = support_rows + query_rows
    width = config.width
    item_count = config.item_summary_count
    summary_count = config.feature_summary_count
    head_width = config.head_width

    def tensor(rows: int, groups_: int, width_: int = width) -> int:
        return members * rows * groups_ * width_ * element_size

    padded_features = groups * config.columns_per_group
    joined_input = members * joined_rows * padded_features * 4
    encoded_features = tensor(joined_rows, groups)
    encoded_labels = tensor(joined_rows, 1)
    initial_state = tensor(joined_rows, state_groups)
    # ClassificationModel retains the per-member encoder outputs, their
    # stacked copy, label state, and initial transformer state until the nested
    # transformer call returns.
    encoder_retained = (
        joined_input + 2 * encoded_features + encoded_labels + initial_state
    )

    state = tensor(transformer_rows, state_groups)
    item_summary = tensor(transformer_rows, item_count)
    feature_summary = tensor(summary_count, state_groups)
    query_state = tensor(query_rows, state_groups)

    # Packed QKV overlaps the input and SDPA output. When the independent
    # flattened batch is split, completed SDPA pieces overlap their final cat.
    # SDPA pieces and their final cat do not exceed the later projection peak:
    # the temporary list is released before output projection starts.
    attention_output_copies = 1
    support_state = tensor(support_rows, state_groups)
    # Item-context construction holds five additional state-sized storages
    # (the concatenated context, packed Q/K/V, and updated prefix) alongside
    # the support state already counted above. Transposes and head views
    # themselves allocate no storage.
    support_context_overlap = 0 if compact_transformer else 5 * support_state
    # Feature self-attention Q/K/V + SDPA output + residual is the only chunkable
    # transient: rows are a batch axis, so this block shrinks with the row chunk
    # while the resident state / item_summary / item-context floor does not. This
    # is exactly the ``k*C`` term; everything else below is the resident ``R``.
    chunk_rows = min(transformer_rows, feature_attention_row_chunk)
    chunked_feature_tokens = (
        tensor(chunk_rows, state_groups) + tensor(chunk_rows, item_count)
    )
    # An engaged chunk trades that transient for one preallocated full-row
    # destination the blocks are written into (``runtime.run_in_chunks``); the
    # un-chunked path allocates none because its result *is* the block. So the
    # peak floors one destination above the next phase, never on it.
    chunk_destination = (
        0 if chunk_rows >= transformer_rows else state + item_summary
    )
    feature_attention = (
        encoder_retained
        + 2 * state
        # Query projection, residual output, and normalized output overlap the
        # retained joined state in the query-heavy feature-attention phase.
        + 3 * query_state
        + item_summary
        + chunk_destination
        + chunked_feature_tokens
        + 3 * chunked_feature_tokens
        + attention_output_copies * chunked_feature_tokens
        + chunked_feature_tokens
        + support_context_overlap
    )

    context_rows = support_rows + summary_count
    item_context = tensor(context_rows, state_groups)
    query_context = tensor(max(transformer_rows - support_rows, 0), state_groups)
    if transformer_rows == query_rows:
        query_context = tensor(query_rows, state_groups)

    # Item self-attention owns packed QKV and its result while the row-by-group
    # state and summaries remain live.
    item_self_attention = (
        encoder_retained
        + state
        + item_summary
        + feature_summary
        + 6 * item_context
    )

    first_head_context = (
        2
        * members
        * state_groups
        * context_rows
        * head_width
        * element_size
    )
    query_cross_attention = (
        encoder_retained
        + state
        + item_summary
        + feature_summary
        + first_head_context
        + 4 * query_context
    )

    state_tokens = members * transformer_rows * state_groups
    item_summary_tokens = members * transformer_rows * item_count
    feature_summary_tokens = members * summary_count * state_groups
    largest_ffn = max(config.feedforward_widths)
    state_feedforward = (
        encoder_retained
        + item_summary
        + feature_summary
        + _ffn_peak(
            input_bytes=state,
            token_count=state_tokens,
            width=width,
            expanded_width=largest_ffn,
            element_size=element_size,
            feedforward_token_chunk=feedforward_token_chunk,
        )
    )
    summary_feedforwards: tuple[int, ...] = ()
    if config.block_count > 1:
        summary_ffn = max(config.feedforward_widths[:-1])
        summary_feedforwards = (
            encoder_retained
            + state
            + item_summary
            + _ffn_peak(
                input_bytes=feature_summary,
                token_count=feature_summary_tokens,
                width=width,
                expanded_width=summary_ffn,
                element_size=element_size,
                feedforward_token_chunk=feedforward_token_chunk,
            ),
            encoder_retained
            + state
            + feature_summary
            + _ffn_peak(
                input_bytes=item_summary,
                token_count=item_summary_tokens,
                width=width,
                expanded_width=summary_ffn,
                element_size=element_size,
                feedforward_token_chunk=feedforward_token_chunk,
            ),
        )
    feedforward = max((state_feedforward, *summary_feedforwards))

    decoder = 0
    if query_rows:
        output_bytes = members * query_rows * output_width * element_size
        # The head flattens all member/query embeddings but receives the outer
        # query size as its chunk bound, so one head call contains at most
        # ``query_rows`` vectors even when several members run together.
        head_chunk_rows = min(query_rows, members * query_rows)
        chunk_input = head_chunk_rows * width * element_size
        chunk_hidden = head_chunk_rows * decoder_hidden_width * element_size
        chunk_output = head_chunk_rows * output_width * element_size
        decoder_base = encoder_retained + state + item_summary + feature_summary
        decoder = decoder_base + max(
            2 * output_bytes,
            output_bytes + chunk_input + 2 * chunk_hidden + chunk_output,
        )

    layer_count = config.block_count
    persistent_cache = first_head_context * layer_count if persistent_gpu_cache else 0
    current_transfer = first_head_context if cpu_cache_transfer else 0
    cache_extra = max(persistent_cache, current_transfer)

    # A cache-building layer reaches its working peak while earlier layers'
    # first-head K/V blocks remain resident. CPU offload keeps only the current
    # layer's block on the accelerator.
    feature_attention += cache_extra
    item_self_attention += cache_extra
    query_cross_attention += cache_extra
    feedforward += cache_extra

    result = {
        "encoder": encoder_retained + encoded_features,
        "feature_attention": feature_attention,
        "item_attention": item_self_attention,
        "query_cross_attention": query_cross_attention,
        "feedforward": feedforward,
    }
    if decoder:
        result["decoder"] = decoder
    return result


def feedforward_token_counts(
    config: ModelConfig,
    *,
    support_rows: int,
    query_rows: int,
    feature_count: int,
    members: int,
    mode: PeakMemoryMode | str,
) -> tuple[int, ...]:
    """Return the flattened token counts of the stock transformer's FFNs."""
    if not isinstance(config, ModelConfig):
        raise TypeError("config must be a ModelConfig")
    support_rows = _positive_integer("support_rows", support_rows)
    query_rows = _nonnegative_integer("query_rows", query_rows)
    feature_count = _positive_integer("feature_count", feature_count)
    members = _positive_integer("members", members)
    try:
        resolved_mode = PeakMemoryMode(mode)
    except (TypeError, ValueError) as error:
        raise ValueError("unsupported peak-memory mode") from error

    groups = math.ceil(feature_count / config.columns_per_group)
    state_groups = groups + 1
    support_only = resolved_mode is PeakMemoryMode.SUPPORT_ONLY_BUILD_GPU
    compact = resolved_mode in {
        PeakMemoryMode.CACHED_COMPACT_GPU,
        PeakMemoryMode.CACHED_COMPACT_CPU,
    }
    transformer_rows = (
        support_rows
        if support_only
        else query_rows
        if compact
        else support_rows + query_rows
    )
    counts: list[int] = []
    for _ in range(config.block_count - 1):
        counts.extend(
            (
                members * transformer_rows * state_groups,
                members * config.feature_summary_count * state_groups,
                members * transformer_rows * config.item_summary_count,
            )
        )
    if not support_only and query_rows:
        # The stock final layer materializes only the query label-group FFN.
        counts.append(members * query_rows)
    return tuple(counts)


def estimated_peak(
    config: ModelConfig,
    *,
    support_rows: int,
    query_rows: int,
    feature_count: int,
    members: int,
    feedforward_token_chunk: int,
    dtype: torch.dtype,
    mode: PeakMemoryMode | str,
    max_vram_bytes: int,
    feature_attention_row_chunk: int = FULL_FEATURE_ATTENTION_ROWS,
    baseline_allocated_bytes: int = 0,
    baseline_reserved_bytes: int = 0,
    free_bytes: int | None = None,
    output_width: int | None = None,
    decoder_hidden_width: int | None = None,
    outer_output_bytes: int = 0,
) -> PeakMemoryEstimate:
    """Estimate one inference phase without allocating shape-sized tensors.

    ``allocated_bytes`` and ``projected_reserved_bytes`` are process totals,
    including the supplied allocator baseline.  ``free_bytes`` is optional so
    the same pure calculation can be used in CPU unit tests; CUDA callers pass
    ``torch.cuda.mem_get_info(device)[0]`` to include external contention.
    """
    if not isinstance(config, ModelConfig):
        raise TypeError("config must be a ModelConfig")
    support_rows = _positive_integer("support_rows", support_rows)
    query_rows = _nonnegative_integer("query_rows", query_rows)
    feature_count = _positive_integer("feature_count", feature_count)
    members = _positive_integer("members", members)
    feedforward_token_chunk = _positive_integer(
        "feedforward_token_chunk", feedforward_token_chunk
    )
    feature_attention_row_chunk = _positive_integer(
        "feature_attention_row_chunk", feature_attention_row_chunk
    )
    max_vram_bytes = _positive_integer("max_vram_bytes", max_vram_bytes)
    baseline_allocated_bytes = _nonnegative_integer(
        "baseline_allocated_bytes", baseline_allocated_bytes
    )
    baseline_reserved_bytes = _nonnegative_integer(
        "baseline_reserved_bytes", baseline_reserved_bytes
    )
    outer_output_bytes = _nonnegative_integer(
        "outer_output_bytes", outer_output_bytes
    )
    output_width = _positive_integer(
        "output_width",
        config.class_capacity if output_width is None else output_width,
    )
    decoder_hidden_width = _positive_integer(
        "decoder_hidden_width",
        (
            config.feedforward_widths[-1]
            if decoder_hidden_width is None
            else decoder_hidden_width
        ),
    )
    if baseline_reserved_bytes < baseline_allocated_bytes:
        raise ValueError("reserved baseline cannot be smaller than allocated baseline")
    if free_bytes is not None:
        free_bytes = _nonnegative_integer("free_bytes", free_bytes)
    try:
        resolved_mode = PeakMemoryMode(mode)
    except (TypeError, ValueError) as error:
        raise ValueError("unsupported peak-memory mode") from error
    element_size = _element_size(dtype)

    compact = resolved_mode in {
        PeakMemoryMode.CACHED_COMPACT_GPU,
        PeakMemoryMode.CACHED_COMPACT_CPU,
    }
    support_only = resolved_mode is PeakMemoryMode.SUPPORT_ONLY_BUILD_GPU
    transformer_rows = (
        support_rows
        if support_only
        else query_rows
        if compact
        else support_rows + query_rows
    )
    persistent_gpu_cache = resolved_mode in {
        PeakMemoryMode.CACHE_BUILD_GPU,
        PeakMemoryMode.CACHED_PADDED_GPU,
        PeakMemoryMode.CACHED_COMPACT_GPU,
        PeakMemoryMode.SUPPORT_ONLY_BUILD_GPU,
    }
    cpu_cache_transfer = resolved_mode in {
        PeakMemoryMode.CACHE_BUILD_CPU,
        PeakMemoryMode.CACHED_PADDED_CPU,
        PeakMemoryMode.CACHED_COMPACT_CPU,
    }
    phases = _working_set(
        config,
        support_rows=support_rows,
        query_rows=0 if support_only else query_rows,
        feature_count=feature_count,
        members=members,
        element_size=element_size,
        transformer_rows=transformer_rows,
        compact_transformer=compact,
        persistent_gpu_cache=persistent_gpu_cache,
        cpu_cache_transfer=cpu_cache_transfer,
        feedforward_token_chunk=feedforward_token_chunk,
        feature_attention_row_chunk=feature_attention_row_chunk,
        output_width=output_width,
        decoder_hidden_width=decoder_hidden_width,
    )
    if outer_output_bytes:
        # Query/member tiles are accumulated by the shared executor. During a
        # model call, at most one full-output equivalent is retained outside
        # the call. The final concatenation instead overlaps source tiles and
        # their destination without transformer working tensors.
        phases = {
            name: value + outer_output_bytes
            for name, value in phases.items()
        }
        phases["output_concatenation"] = 2 * outer_output_bytes
    limiting_phase, incremental_allocated = max(phases.items(), key=lambda item: item[1])
    allocated_total = baseline_allocated_bytes + incremental_allocated

    reusable = baseline_reserved_bytes - baseline_allocated_bytes
    new_live_bytes = max(0, incremental_allocated - reusable)
    groups = math.ceil(feature_count / config.columns_per_group)
    state_groups = groups + 1
    state_tokens = members * transformer_rows * state_groups
    item_summary_tokens = members * transformer_rows * config.item_summary_count
    feature_summary_tokens = (
        members * config.feature_summary_count * state_groups
    )
    ffn_tokens = min(
        max(state_tokens, item_summary_tokens, feature_summary_tokens),
        feedforward_token_chunk,
    )
    ffn_reservation_tail = (
        ffn_tokens
        * (2 * max(config.feedforward_widths) + config.width)
        * element_size
    )
    # Allocator segments from the FFN expansion can remain reserved when a
    # later attention phase reaches the allocated peak. Account for that
    # phase-history tail, or a 6% large-block tail when it is larger.
    reservation_tail = max(
        (new_live_bytes * 6 + 99) // 100,
        (ffn_reservation_tail * 5 + 3) // 4,
    )
    if resolved_mode is PeakMemoryMode.CACHE_BUILD_GPU:
        context_rows = support_rows + config.feature_summary_count
        persistent_cache = (
            2
            * config.block_count
            * members
            * state_groups
            * context_rows
            * config.head_width
            * element_size
        )
        # Live per-layer KV blocks pin allocator segments while later large
        # workspaces come and go during a joined cache build.
        reservation_tail += (persistent_cache * 3 + 1) // 2
    elif resolved_mode is PeakMemoryMode.CACHE_BUILD_CPU:
        context_rows = support_rows + config.feature_summary_count
        current_transfer = (
            2
            * members
            * state_groups
            * context_rows
            * config.head_width
            * element_size
        )
        # The just-offloaded GPU K/V pair can leave allocator segments behind
        # while the next layer starts, even though only one pair is allocated.
        reservation_tail += (current_transfer * 5 + 7) // 8
    new_reserved = _round_up(
        new_live_bytes + reservation_tail,
        _CUDA_LARGE_BLOCK_BYTES,
    )
    projected_reserved = baseline_reserved_bytes + new_reserved
    physically_fits = free_bytes is None or new_reserved <= free_bytes
    return PeakMemoryEstimate(
        allocated_bytes=int(allocated_total),
        projected_reserved_bytes=int(projected_reserved),
        target_vram_bytes=max_vram_bytes,
        limiting_phase=limiting_phase,
        fits=projected_reserved <= max_vram_bytes and physically_fits,
    )


__all__ = ["PeakMemoryEstimate", "PeakMemoryMode", "estimated_peak"]
