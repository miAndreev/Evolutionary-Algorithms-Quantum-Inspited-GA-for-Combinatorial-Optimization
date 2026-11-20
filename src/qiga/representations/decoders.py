"""Decoders map an observed bit string (a measured Q-chromosome) to a problem solution.

* :class:`BinaryDecoder` - one Q-bit per gene, the bits *are* the solution (knapsack).
* :class:`RandomKeyDecoder` - ``bits_per_key`` Q-bits per element; each group of bits is read
  as an unsigned integer key (most significant bit first) and the permutation is the
  elements sorted by key (Bean, 1994). Equal keys keep their index order, so decoding is
  deterministic: the same bits always give the same tour.

``encode`` is the inverse direction, used to write a repaired solution back into bits.
"""

from abc import ABC
from abc import abstractmethod
import math
from typing import TYPE_CHECKING

import numpy as np

from qiga.problems.base import Encoding

if TYPE_CHECKING:
    from qiga.problems.base import Problem
    from qiga.problems.base import Solution

MAX_BITS_PER_KEY = 30


class Decoder(ABC):
    """Bits <-> solution for a problem of ``dimension`` genes."""

    def __init__(self, dimension: int) -> None:
        if dimension < 1:
            msg = "dimension must be a positive integer"
            raise ValueError(msg)
        self.dimension = dimension

    @property
    @abstractmethod
    def n_bits(self) -> int:
        """Number of Q-bits per chromosome."""

    @abstractmethod
    def decode(self, bits: Solution) -> list[int]:
        """Solution encoded by ``bits``."""

    @abstractmethod
    def encode(self, solution: Solution) -> np.ndarray:
        """Bits (int8) that decode to ``solution``."""

    def _check_bits(self, bits: Solution) -> np.ndarray:
        array = np.asarray(bits)
        if array.shape != (self.n_bits,):
            msg = f"expected {self.n_bits} bits, got shape {array.shape}"
            raise ValueError(msg)
        return array.astype(np.int64)


class BinaryDecoder(Decoder):
    @property
    def n_bits(self) -> int:
        return self.dimension

    def decode(self, bits: Solution) -> list[int]:
        return self._check_bits(bits).tolist()

    def encode(self, solution: Solution) -> np.ndarray:
        return self._check_bits(solution).astype(np.int8)


class RandomKeyDecoder(Decoder):
    """Permutation of ``range(dimension)`` from ``dimension * bits_per_key`` bits."""

    def __init__(self, dimension: int, bits_per_key: int) -> None:
        super().__init__(dimension)
        if not 1 <= bits_per_key <= MAX_BITS_PER_KEY:
            msg = f"bits_per_key must be in [1, {MAX_BITS_PER_KEY}], got {bits_per_key}"
            raise ValueError(msg)
        self.bits_per_key = bits_per_key
        self._weights = 1 << np.arange(bits_per_key - 1, -1, -1, dtype=np.int64)

    @staticmethod
    def default_bits_per_key(dimension: int) -> int:
        """Enough bits to give every element a distinct key, plus one to make ties rarer."""
        return min(max(math.ceil(math.log2(dimension)), 1) + 1, MAX_BITS_PER_KEY)

    @property
    def n_bits(self) -> int:
        return self.dimension * self.bits_per_key

    def keys(self, bits: Solution) -> np.ndarray:
        """Integer key of every element."""
        return self._check_bits(bits).reshape(self.dimension, self.bits_per_key) @ self._weights

    def decode(self, bits: Solution) -> list[int]:
        return np.argsort(self.keys(bits), kind="stable").tolist()

    def encode(self, solution: Solution) -> np.ndarray:
        """Keys spread evenly over the key range, in tour order. Needs ``2**bits_per_key >= dimension``."""
        tour = np.asarray(solution)
        if not np.array_equal(np.sort(tour), np.arange(self.dimension)):
            msg = f"solution must be a permutation of range({self.dimension})"
            raise ValueError(msg)
        top = (1 << self.bits_per_key) - 1
        if top + 1 < self.dimension:
            msg = f"{self.bits_per_key} bits per key cannot give {self.dimension} distinct keys"
            raise ValueError(msg)
        keys = np.empty(self.dimension, dtype=np.int64)
        keys[tour] = np.arange(self.dimension) * top // max(self.dimension - 1, 1)
        return ((keys[:, None] & self._weights) > 0).astype(np.int8).ravel()


def make_decoder(problem: Problem, bits_per_key: int | None = None) -> Decoder:
    """The natural decoder for the problem's encoding."""
    if problem.encoding is Encoding.BINARY:
        return BinaryDecoder(problem.dimension)
    if problem.encoding is Encoding.PERMUTATION:
        return RandomKeyDecoder(problem.dimension, bits_per_key or RandomKeyDecoder.default_bits_per_key(problem.dimension))
    msg = f"no decoder for {problem.encoding} problems"  # pragma: no cover - exhaustive over Encoding
    raise ValueError(msg)  # pragma: no cover
