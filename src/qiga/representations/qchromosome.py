"""A string of Q-bits stored as two amplitude vectors."""

from typing import TYPE_CHECKING
from typing import Self

import numpy as np

from qiga.representations import gates
from qiga.representations.qbit import QBit

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


class QChromosome:
    """``dimension`` Q-bits; observing it gives a 0/1 vector of that length.

    Create it either from a ``dimension`` (all Q-bits in equal superposition) or from an
    ``(n, 2)`` array of ``[alpha, beta]`` rows, which is normalized row by row.
    All gates modify the chromosome in place; use :meth:`copy` to keep the original.
    """

    __slots__ = ("_alpha", "_beta")

    def __init__(self, dimension: int | None = None, amplitudes: ArrayLike | None = None) -> None:
        if amplitudes is None:
            if dimension is None or dimension < 1:
                msg = "dimension must be a positive integer"
                raise ValueError(msg)
            self._alpha = np.full(dimension, gates.INV_SQRT2)
            self._beta = np.full(dimension, gates.INV_SQRT2)
            return
        array = np.asarray(amplitudes, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != 2 or array.shape[0] < 1:
            msg = f"amplitudes must have shape (n, 2) with n >= 1, got {array.shape}"
            raise ValueError(msg)
        if dimension is not None and dimension != array.shape[0]:
            msg = f"dimension {dimension} does not match {array.shape[0]} amplitude rows"
            raise ValueError(msg)
        self._alpha, self._beta = gates.normalize(array[:, 0], array[:, 1])

    @classmethod
    def from_probabilities(cls, p_one: ArrayLike) -> Self:
        """Q-bits with non-negative amplitudes and the given P(observe 1)."""
        p = np.asarray(p_one, dtype=np.float64)
        if p.ndim != 1 or ((p < 0) | (p > 1)).any():
            msg = "p_one must be a 1-D array of probabilities"
            raise ValueError(msg)
        return cls(amplitudes=np.column_stack([np.sqrt(1.0 - p), np.sqrt(p)]))

    @property
    def dimension(self) -> int:
        return self._alpha.shape[0]

    def __len__(self) -> int:
        return self.dimension

    def __getitem__(self, index: int) -> QBit:
        """A copy of one Q-bit."""
        return QBit(self._alpha[index], self._beta[index])

    @property
    def alpha(self) -> np.ndarray:
        return self._readonly(self._alpha)

    @property
    def beta(self) -> np.ndarray:
        return self._readonly(self._beta)

    @property
    def amplitudes(self) -> np.ndarray:
        """``(n, 2)`` array of ``[alpha, beta]`` rows (a copy)."""
        return np.column_stack([self._alpha, self._beta])

    def probabilities(self) -> np.ndarray:
        """``(n, 2)`` array of ``[P(0), P(1)]`` rows."""
        return np.square(self.amplitudes)

    def p_one(self) -> np.ndarray:
        """P(observe 1) for every Q-bit."""
        return np.square(self._beta)

    def measure(self, rng: np.random.Generator) -> np.ndarray:
        """Observe all Q-bits; returns an int8 0/1 vector. The chromosome itself is unchanged."""
        return gates.measure(self._beta, rng)

    def rotate(self, theta: ArrayLike) -> None:
        """Rotate every Q-bit by ``theta`` radians (a scalar, or one angle per Q-bit)."""
        theta = np.asarray(theta, dtype=np.float64)
        if theta.ndim > 1 or (theta.ndim == 1 and theta.shape[0] != self.dimension):
            msg = f"theta must be a scalar or have length {self.dimension}, got shape {theta.shape}"
            raise ValueError(msg)
        self._alpha, self._beta = gates.normalize(*gates.rotate(self._alpha, self._beta, theta))

    def apply_not(self, mask: ArrayLike | None = None) -> None:
        """Swap alpha and beta of every Q-bit, or only where ``mask`` is true."""
        if mask is not None and np.shape(mask) != (self.dimension,):
            msg = f"mask must have length {self.dimension}"
            raise ValueError(msg)
        self._alpha, self._beta = gates.quantum_not(self._alpha, self._beta, mask)

    def apply_h_epsilon(self, epsilon: float) -> None:
        """Keep every P(observe 1) inside [epsilon, 1 - epsilon]."""
        self._alpha, self._beta = gates.h_epsilon(self._alpha, self._beta, epsilon)

    def copy(self) -> Self:
        clone = object.__new__(type(self))
        clone._alpha, clone._beta = self._alpha.copy(), self._beta.copy()
        return clone

    def __repr__(self) -> str:
        return f"QChromosome(dimension={self.dimension}, mean_p_one={self.p_one().mean():.3f})"

    @staticmethod
    def _readonly(array: np.ndarray) -> np.ndarray:
        view = array.view()
        view.flags.writeable = False
        return view
