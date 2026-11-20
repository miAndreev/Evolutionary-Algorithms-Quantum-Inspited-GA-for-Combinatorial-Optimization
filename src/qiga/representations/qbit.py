"""A single Q-bit. Readable reference for the gates; QIGA itself uses :class:`QChromosome`."""

import numpy as np

from qiga.representations import gates


class QBit:
    """Q-bit with real amplitudes, normalized on construction (default: equal superposition)."""

    __slots__ = ("_alpha", "_beta")

    def __init__(self, alpha: float = gates.INV_SQRT2, beta: float = gates.INV_SQRT2) -> None:
        a, b = gates.normalize(alpha, beta)
        self._alpha, self._beta = float(a), float(b)

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def beta(self) -> float:
        return self._beta

    def probabilities(self) -> tuple[float, float]:
        """(P(observe 0), P(observe 1))."""
        return self._alpha**2, self._beta**2

    def measure(self, rng: np.random.Generator) -> int:
        return int(gates.measure(np.array(self._beta), rng))

    def rotate(self, theta: float) -> None:
        self._set(*gates.rotate(np.array(self._alpha), np.array(self._beta), theta))

    def apply_not(self) -> None:
        self._alpha, self._beta = self._beta, self._alpha

    def apply_h_epsilon(self, epsilon: float) -> None:
        self._set(*gates.h_epsilon(np.array(self._alpha), np.array(self._beta), epsilon))

    def _set(self, alpha: np.ndarray, beta: np.ndarray) -> None:
        a, b = gates.normalize(alpha, beta)
        self._alpha, self._beta = float(a), float(b)

    def __repr__(self) -> str:
        return f"QBit(alpha={self._alpha:.4f}, beta={self._beta:.4f})"
