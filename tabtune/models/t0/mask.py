"""Attention masks built from per-patch metadata, shared by every runtime.

Naming: a ``patched_`` prefix marks an array reshaped into patches but still per-time-step,
``(V, P, patch_size)``; a ``patch_`` prefix marks one value per patch, ``(V, P)``.
Convention: ``True`` = can attend, ``False`` = blocked (as scaled-dot-product attention takes it).
"""

from typing import Any

from jaxtyping import Bool, Int

from ._ops import ArrayT, ops_for
from .types import MaskType, VariateType

__all__ = ["build_attention_masks", "reduce_patch_metadata"]


def reduce_patch_metadata(
    patched_metadata: Int[ArrayT, "variates patches patch_size"],
    patched_mask: Int[Any, "variates patches patch_size"],
) -> Int[ArrayT, "variates patches"]:
    """Collapse patched metadata to each patch's first non-PAD time step, or ``-1`` when fully PAD.

    Args:
        patched_metadata: Patched ``group_ids`` or ``variate_type``.
        patched_mask: Patched typed mask, the authority on which time steps are padding.

    Returns:
        One value per patch.
    """
    ops = ops_for(patched_metadata)
    is_real = patched_mask != MaskType.PAD
    first_real = ops.argmax(ops.astype(is_real, ops.int32), axis=-1, keepdims=True)
    reduced = ops.squeeze(ops.take_along_axis(patched_metadata, first_real, axis=-1), axis=-1)
    return ops.where(
        ops.any(is_real, axis=-1), reduced, ops.full((), -1, dtype=reduced.dtype, device=ops.device(reduced))
    )


def build_attention_masks(
    patched_group_ids: Int[ArrayT, "variates patches patch_size"],
    patched_variate_type: Int[ArrayT, "variates patches patch_size"],
    patched_mask: Int[ArrayT, "variates patches patch_size"],
) -> tuple[Bool[ArrayT, "variates 1 patches patches"], Bool[ArrayT, "patches 1 variates variates"]]:
    """The two masks of one forward pass.

    - **per-variate time mask** ``(V, 1, P, P)`` — causal for target / historical
      variates, bidirectional for futures; a patch attends only patches of its own
      series, and no query reads an all-PAD patch.
    - **per-patch group mask** ``(P, 1, V, V)`` — at each patch, only variates that
      share a group id attend to one another, and a future-covariate query reads
      future-covariate keys only, so information flows from covariates into targets and
      never back. The diagonal stays valid, so every real query keeps a key.
    """
    ops = ops_for(patched_group_ids)
    patch_group_ids = reduce_patch_metadata(patched_group_ids, patched_mask)
    patch_variate_type = reduce_patch_metadata(patched_variate_type, patched_mask)
    attendable = ops.any(patched_mask != MaskType.PAD, axis=-1)
    valid = patch_group_ids >= 0
    is_future = patch_variate_type == VariateType.FUTURE

    same_series = (
        (ops.expand_dims(patch_group_ids, axis=2) == ops.expand_dims(patch_group_ids, axis=1))
        & ops.expand_dims(valid, axis=2)
        & ops.expand_dims(valid, axis=1)
    )
    positions = ops.arange(patch_group_ids.shape[1], device=ops.device(patch_group_ids))
    causal = ops.expand_dims(positions, axis=1) >= ops.expand_dims(positions, axis=0)
    # Boolean logic rather than a boolean `where`, which some ONNX runtimes (ORT CPU/WASM) lack.
    future_query = ops.expand_dims(is_future, axis=2)
    time_mask = (future_query & same_series) | (~future_query & same_series & causal)
    time_mask = time_mask & ops.expand_dims(attendable, axis=1)

    ids_by_patch, valid_by_patch, future_by_patch = patch_group_ids.T, valid.T, is_future.T
    group_mask = (
        (ops.expand_dims(ids_by_patch, axis=2) == ops.expand_dims(ids_by_patch, axis=1))
        & ops.expand_dims(valid_by_patch, axis=2)
        & ops.expand_dims(valid_by_patch, axis=1)
    )
    future_reads_non_future = ops.expand_dims(future_by_patch, axis=2) & ~ops.expand_dims(future_by_patch, axis=1)
    group_mask = group_mask & ~future_reads_non_future
    return ops.expand_dims(time_mask, axis=1), ops.expand_dims(group_mask, axis=1)
