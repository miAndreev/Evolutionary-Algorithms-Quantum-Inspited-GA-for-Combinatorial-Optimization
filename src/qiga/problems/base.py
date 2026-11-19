"""Common interface for all optimization problems."""

from abc import ABC
from abc import abstractmethod
from collections.abc import Sequence
from enum import StrEnum
from typing import ClassVar

import numpy as np
from pydantic import BaseModel
from pydantic import ConfigDict


class Sense(StrEnum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class Encoding(StrEnum):
    """How a candidate solution is represented; algorithms pick operators by this."""

    BINARY = "binary"
    PERMUTATION = "permutation"


type Solution = Sequence[int] | np.ndarray


class Problem(BaseModel, ABC):
    """An immutable, validated problem instance.

    Malformed solutions (wrong length, non-integer genes, values outside the
    encoding's domain) raise ``ValueError``. Constraint violations are reported
    by :meth:`is_feasible` and handled by :meth:`evaluate` per problem.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    sense: ClassVar[Sense]
    encoding: ClassVar[Encoding]

    name: str | None = None

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Length of a solution vector."""

    @abstractmethod
    def evaluate(self, solution: Solution) -> float:
        """Objective value of ``solution``."""

    @abstractmethod
    def is_feasible(self, solution: Solution) -> bool:
        """Whether ``solution`` satisfies the problem constraints."""

    @abstractmethod
    def random_solution(self, rng: np.random.Generator) -> list[int]:
        """Uniformly random solution in the encoding's domain (not necessarily feasible)."""

    def is_better(self, a: float, b: float) -> bool:
        """Whether objective value ``a`` is strictly better than ``b``."""
        return a > b if self.sense is Sense.MAXIMIZE else a < b

    def _as_array(self, solution: Solution) -> np.ndarray:
        array = np.asarray(solution)
        if array.dtype == np.bool_:
            array = array.astype(np.int64)
        if array.ndim != 1 or array.shape[0] != self.dimension:
            msg = f"solution must be a 1-D sequence of length {self.dimension}, got shape {array.shape}"
            raise ValueError(msg)
        if not np.issubdtype(array.dtype, np.integer):
            msg = f"solution genes must be integers, got dtype {array.dtype}"
            raise ValueError(msg)
        return array
