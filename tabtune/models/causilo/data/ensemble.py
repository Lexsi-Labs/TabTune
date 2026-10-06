"""Deterministic input permutations; all ensemble members share model weights."""

import math
import random
from dataclasses import dataclass

METHOD_ORDER = ("none", "rank2gaussian", "robust", "power")


def _circular_distances(step: int, width: int) -> tuple[int, ...]:
    """Distances to the first two neighbors when walking a ring with this stride."""
    return tuple(
        min((multiple * step) % width, (-multiple * step) % width)
        for multiple in (1, 2)
    )


def make_affine_orders(width: int, count: int, random_state: int) -> list[tuple[int, ...]]:
    """Build feature permutations while discouraging repeated local neighborhoods.

    Coprime strides visit every feature exactly once. Among at most 32 candidate
    strides, prefer the least-used distances to the first two ring neighbors.
    Keep the first member in identity order. Separate local RNG streams preserve
    the seeded policy without changing Python's global random state.
    """
    if width < 2:
        return [tuple(range(width)) for _ in range(count)]
    generator = random.Random(random_state)
    ring = random.Random(random_state).sample(range(width), width)
    strides = [step for step in range(1, width) if math.gcd(step, width) == 1]
    usage = {}
    orders = []
    for index in range(count):
        candidates = generator.sample(strides, min(32, len(strides))) if index else [1]
        step = min(
            candidates,
            key=lambda stride: sum(usage.get(d, 0) for d in _circular_distances(stride, width)),
        )
        offset = generator.randrange(width) if index else 0
        if index == 0:
            order = tuple(range(width))
        else:
            order = tuple(ring[(offset + step * column) % width] for column in range(width))
        orders.append(order)
        for distance in _circular_distances(step, width):
            usage[distance] = usage.get(distance, 0) + 1
    return orders


@dataclass(frozen=True)
class EnsembleMember:
    """One normalized/permuted input view of the shared model.

    ``feature_order`` indexes the encoded, nonconstant feature table.
    ``class_order[original_id]`` is the permuted class ID used as a target and
    the model output column to read when restoring original class order.
    Regression members have no class permutation.
    """

    normalization: str
    feature_order: tuple[int, ...]
    class_order: tuple[int, ...] | None


def make_ensemble_members(
    width: int, classes: int, count: int, random_state: int
) -> tuple[EnsembleMember, ...]:
    """Cycle normalizers and permutations in construction order, fixed by the seed."""
    if count == 1:
        return (EnsembleMember("none", tuple(range(width)), tuple(range(classes)) if classes else None),)
    label_generator = random.Random(random_state)
    class_orders = []
    for index in range(count):
        if not classes:
            class_orders.append(None)
            continue
        if index % classes == 0:
            # Rotate each sampled class ordering so labels occupy every slot
            # before another base ordering is drawn.
            base = label_generator.sample(range(classes), classes)
        offset = index % classes
        class_orders.append(tuple(base[(label + offset) % classes] for label in range(classes)))
    feature_orders = make_affine_orders(width, count, random_state)
    return tuple(
        EnsembleMember(METHOD_ORDER[i % 4], feature_orders[i], class_orders[i])
        for i in range(count)
    )
