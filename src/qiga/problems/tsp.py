"""Travelling Salesman Problem (closed tour, possibly asymmetric)."""

from functools import cached_property
from typing import TYPE_CHECKING
from typing import ClassVar
from typing import Self

import numpy as np
from pydantic import Field
from pydantic import NonNegativeFloat
from pydantic import model_validator

from qiga.problems.base import Encoding
from qiga.problems.base import Problem
from qiga.problems.base import Sense
from qiga.problems.base import Solution

if TYPE_CHECKING:
    from collections.abc import Sequence


class TSP(Problem):
    """Minimize the length of a closed tour visiting every city exactly once.

    A solution is a permutation of ``range(n)``; the tour returns to its first city.
    ``coordinates`` is optional and only used for visualization.
    """

    sense: ClassVar[Sense] = Sense.MINIMIZE
    encoding: ClassVar[Encoding] = Encoding.PERMUTATION

    distances: tuple[tuple[NonNegativeFloat, ...], ...] = Field(min_length=2)
    coordinates: tuple[tuple[float, float], ...] | None = None

    @model_validator(mode="after")
    def _check_matrix(self) -> Self:
        n = len(self.distances)
        if any(len(row) != n for row in self.distances):
            msg = "distance matrix must be square"
            raise ValueError(msg)
        if any(self.distances[i][i] != 0 for i in range(n)):
            msg = "distance matrix diagonal must be zero"
            raise ValueError(msg)
        if self.coordinates is not None and len(self.coordinates) != n:
            msg = f"expected {n} coordinates, got {len(self.coordinates)}"
            raise ValueError(msg)
        return self

    @cached_property
    def _distances(self) -> np.ndarray:
        return np.array(self.distances, dtype=np.float64)

    @classmethod
    def from_coordinates(cls, coordinates: Sequence[Sequence[float]], *, rounded: bool = False, name: str | None = None) -> Self:
        """Build a Euclidean instance. ``rounded`` applies TSPLIB EUC_2D rounding to the nearest integer."""
        points = np.asarray(coordinates, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2:
            msg = f"coordinates must have shape (n, 2), got {points.shape}"
            raise ValueError(msg)
        distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
        if rounded:
            distances = np.floor(distances + 0.5)
        return cls(distances=distances.tolist(), coordinates=points.tolist(), name=name)

    @property
    def dimension(self) -> int:
        return len(self.distances)

    def is_feasible(self, solution: Solution) -> bool:
        route = self._as_array(solution)
        return bool(np.array_equal(np.sort(route), np.arange(self.dimension)))

    def evaluate(self, solution: Solution) -> float:
        if not self.is_feasible(solution):
            msg = f"TSP solution must be a permutation of range({self.dimension})"
            raise ValueError(msg)
        route = np.asarray(solution)
        return float(self._distances[route, np.roll(route, -1)].sum())

    def random_solution(self, rng: np.random.Generator) -> list[int]:
        return rng.permutation(self.dimension).tolist()
