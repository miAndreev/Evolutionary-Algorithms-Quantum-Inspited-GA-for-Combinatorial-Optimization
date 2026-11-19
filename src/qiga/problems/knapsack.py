"""0/1 Knapsack problem."""

from functools import cached_property
from typing import ClassVar
from typing import Self

import numpy as np
from pydantic import Field
from pydantic import NonNegativeInt
from pydantic import PositiveInt
from pydantic import model_validator

from qiga.problems.base import Encoding
from qiga.problems.base import Problem
from qiga.problems.base import Sense
from qiga.problems.base import Solution


class Knapsack(Problem):
    """Maximize total value of selected items without exceeding ``capacity``.

    Infeasible solutions evaluate to 0 (death penalty). Use :meth:`repair`
    to turn any bit string into a feasible one.
    """

    sense: ClassVar[Sense] = Sense.MAXIMIZE
    encoding: ClassVar[Encoding] = Encoding.BINARY

    weights: tuple[PositiveInt, ...] = Field(min_length=1)
    values: tuple[PositiveInt, ...] = Field(min_length=1)
    capacity: NonNegativeInt

    @model_validator(mode="after")
    def _check_lengths(self) -> Self:
        if len(self.weights) != len(self.values):
            msg = f"weights ({len(self.weights)}) and values ({len(self.values)}) must have the same length"
            raise ValueError(msg)
        return self

    @cached_property
    def _weights(self) -> np.ndarray:
        return np.array(self.weights, dtype=np.int64)

    @cached_property
    def _values(self) -> np.ndarray:
        return np.array(self.values, dtype=np.int64)

    @cached_property
    def _by_ratio(self) -> np.ndarray:
        """Item indices ordered from worst to best value/weight ratio."""
        return np.argsort(self._values / self._weights, kind="stable")

    @property
    def dimension(self) -> int:
        return len(self.weights)

    def total_weight(self, solution: Solution) -> int:
        return int(self._weights @ self._bits(solution))

    def total_value(self, solution: Solution) -> int:
        return int(self._values @ self._bits(solution))

    def is_feasible(self, solution: Solution) -> bool:
        return self.total_weight(solution) <= self.capacity

    def evaluate(self, solution: Solution) -> float:
        bits = self._bits(solution)
        if self._weights @ bits > self.capacity:
            return 0.0
        return float(self._values @ bits)

    def random_solution(self, rng: np.random.Generator) -> list[int]:
        return rng.integers(0, 2, size=self.dimension).tolist()

    def repair(self, solution: Solution) -> list[int]:
        """Greedy repair: drop worst-ratio items until feasible, then add best-ratio items that fit."""
        bits = self._bits(solution).copy()
        weight = int(self._weights @ bits)
        for i in self._by_ratio:
            if weight <= self.capacity:
                break
            if bits[i]:
                bits[i] = 0
                weight -= int(self._weights[i])
        for i in self._by_ratio[::-1]:
            if not bits[i] and weight + self._weights[i] <= self.capacity:
                bits[i] = 1
                weight += int(self._weights[i])
        return bits.tolist()

    def exact_optimum(self) -> int:
        """Optimal value by dynamic programming, O(n * capacity) time and O(capacity) memory."""
        best = np.zeros(self.capacity + 1, dtype=np.int64)  # best[c] = max value with weight <= c
        for weight, value in zip(self._weights, self._values, strict=True):
            if weight <= self.capacity:
                best[weight:] = np.maximum(best[weight:], best[: self.capacity + 1 - weight] + value)
        return int(best[-1])

    def _bits(self, solution: Solution) -> np.ndarray:
        bits = self._as_array(solution)
        if not np.isin(bits, (0, 1)).all():
            msg = "knapsack solution genes must be 0 or 1"
            raise ValueError(msg)
        return bits
