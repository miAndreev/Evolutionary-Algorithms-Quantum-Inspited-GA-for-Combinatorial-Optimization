"""Reproducible benchmark instance generators."""

from enum import StrEnum

import numpy as np

from qiga.problems.knapsack import Knapsack
from qiga.problems.tsp import TSP


class KnapsackKind(StrEnum):
    """Pisinger's classic correlation classes; harder as correlation grows."""

    UNCORRELATED = "uncorrelated"  # v ~ U[1, R]
    WEAKLY_CORRELATED = "weakly_correlated"  # v ~ U[w - R/10, w + R/10], v >= 1
    STRONGLY_CORRELATED = "strongly_correlated"  # v = w + R/10


def generate_knapsack(
    n: int,
    kind: KnapsackKind = KnapsackKind.UNCORRELATED,
    *,
    max_weight: int = 1000,
    capacity_ratio: float = 0.5,
    seed: int | None = None,
) -> Knapsack:
    """Random instance with weights ~ U[1, max_weight] and capacity = capacity_ratio * sum(weights)."""
    if n < 1:
        msg = "n must be positive"
        raise ValueError(msg)
    if not 0 < capacity_ratio < 1:
        msg = "capacity_ratio must be in (0, 1)"
        raise ValueError(msg)
    rng = np.random.default_rng(seed)
    weights = rng.integers(1, max_weight + 1, size=n)
    spread = max(max_weight // 10, 1)
    match KnapsackKind(kind):
        case KnapsackKind.UNCORRELATED:
            values = rng.integers(1, max_weight + 1, size=n)
        case KnapsackKind.WEAKLY_CORRELATED:
            values = np.maximum(weights + rng.integers(-spread, spread + 1, size=n), 1)
        case KnapsackKind.STRONGLY_CORRELATED:
            values = weights + spread
    capacity = int(capacity_ratio * weights.sum())
    return Knapsack(
        weights=weights.tolist(),
        values=values.tolist(),
        capacity=capacity,
        name=f"knapsack-{KnapsackKind(kind).value}-n{n}-s{seed}",
    )


def generate_euclidean_tsp(n: int, *, size: float = 1000.0, rounded: bool = False, seed: int | None = None) -> TSP:
    """Random cities uniformly placed in a ``size`` x ``size`` square."""
    if n < 2:
        msg = "n must be at least 2"
        raise ValueError(msg)
    rng = np.random.default_rng(seed)
    coordinates = rng.uniform(0.0, size, size=(n, 2))
    return TSP.from_coordinates(coordinates, rounded=rounded, name=f"tsp-euclid-n{n}-s{seed}")
