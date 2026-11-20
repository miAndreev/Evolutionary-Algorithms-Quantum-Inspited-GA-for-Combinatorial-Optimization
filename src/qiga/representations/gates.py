"""Q-bit gates as pure, vectorized functions on amplitude arrays.

A Q-bit is a pair of real amplitudes (alpha, beta) with alpha**2 + beta**2 == 1.
Observing it yields 0 with probability alpha**2 and 1 with probability beta**2.
Signs are meaningful: the rotation gate's direction depends on the (alpha, beta) quadrant.
"""

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

INV_SQRT2 = 1.0 / np.sqrt(2.0)

type Amplitudes = tuple[np.ndarray, np.ndarray]


def normalize(alpha: ArrayLike, beta: ArrayLike) -> Amplitudes:
    """Scale each (alpha, beta) pair to unit length; rejects non-finite or all-zero pairs."""
    alpha, beta = np.broadcast_arrays(np.asarray(alpha, dtype=np.float64), np.asarray(beta, dtype=np.float64))
    if not (np.isfinite(alpha).all() and np.isfinite(beta).all()):
        msg = "amplitudes must be finite"
        raise ValueError(msg)
    norm = np.hypot(alpha, beta)
    if (norm == 0).any():
        msg = "a Q-bit needs at least one non-zero amplitude"
        raise ValueError(msg)
    return alpha / norm, beta / norm


def rotate(alpha: np.ndarray, beta: np.ndarray, theta: ArrayLike) -> Amplitudes:
    """Rotation gate U(theta) = [[cos, -sin], [sin, cos]]; positive theta moves probability towards 1."""
    cos, sin = np.cos(theta), np.sin(theta)
    return cos * alpha - sin * beta, sin * alpha + cos * beta


def quantum_not(alpha: np.ndarray, beta: np.ndarray, mask: ArrayLike | None = None) -> Amplitudes:
    """NOT gate: swap alpha and beta (only where ``mask`` is true, if given)."""
    if mask is None:
        return beta.copy(), alpha.copy()
    mask = np.asarray(mask, dtype=bool)
    return np.where(mask, beta, alpha), np.where(mask, alpha, beta)


def h_epsilon(alpha: np.ndarray, beta: np.ndarray, epsilon: float) -> Amplitudes:
    """H_epsilon gate (Han & Kim, 2004): keep each probability inside [epsilon, 1 - epsilon].

    Stops Q-bits from collapsing completely to 0 or 1, so every gene can still flip.
    Signs are preserved; Q-bits already inside the range are untouched.
    """
    if not 0 <= epsilon < 0.5:
        msg = "epsilon must be in [0, 0.5)"
        raise ValueError(msg)
    p_one = beta**2
    clamped = np.clip(p_one, epsilon, 1.0 - epsilon)
    changed = clamped != p_one
    sign_alpha = np.where(alpha < 0, -1.0, 1.0)
    sign_beta = np.where(beta < 0, -1.0, 1.0)
    return (
        np.where(changed, sign_alpha * np.sqrt(1.0 - clamped), alpha),
        np.where(changed, sign_beta * np.sqrt(clamped), beta),
    )


def measure(beta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Observe Q-bits: each becomes 1 with probability beta**2. Returns an int8 array of 0/1."""
    return (rng.random(np.shape(beta)) < np.square(beta)).astype(np.int8)
