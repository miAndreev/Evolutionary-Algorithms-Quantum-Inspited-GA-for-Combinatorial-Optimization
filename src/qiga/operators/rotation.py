"""Rotation-gate update of Han & Kim (2002), "Quantum-inspired evolutionary algorithm for a
class of combinatorial optimization", Table 1.

For every Q-bit i the angle is ``delta(x_i, b_i, f(x) >= f(b)) * s(alpha_i, beta_i)``:

====  ====  ===========  =======  ================  ================  =======  =======
x_i   b_i   f(x)>=f(b)   delta    s: alpha*beta>0   s: alpha*beta<0   alpha=0  beta=0
====  ====  ===========  =======  ================  ================  =======  =======
0     0     any          0        -                 -                 -        -
0     1     false        theta3   +1                -1                0        +-1
0     1     true         theta4   -1                +1                +-1      0
1     0     false        theta5   -1                +1                +-1      0
1     0     true         theta6   +1                -1                0        +-1
1     1     any          0        -                 -                 -        -
====  ====  ===========  =======  ================  ================  =======  =======

The sign columns simply rotate "towards b_i" (rows 3, 5) or "towards x_i" (rows 4, 6),
so the table reduces to :func:`direction_towards` plus two magnitudes:
``delta_worse`` (theta3 = theta5, paper: 0.01*pi) and ``delta_better`` (theta4 = theta6, paper: 0).
"""

import numpy as np


def direction_towards(alpha: np.ndarray, beta: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Sign of the rotation that moves each Q-bit's probability towards ``target`` (0 or 1).

    Returns +1/-1, or 0 when the Q-bit already is the target state.
    """
    product = alpha * beta
    towards_one = np.where(product > 0, 1.0, np.where(product < 0, -1.0, np.where(alpha == 0, 0.0, 1.0)))
    towards_zero = np.where(product > 0, -1.0, np.where(product < 0, 1.0, np.where(beta == 0, 0.0, 1.0)))
    return np.where(np.asarray(target) == 1, towards_one, towards_zero)


def lookup_table_angles(
    alpha: np.ndarray,
    beta: np.ndarray,
    observed: np.ndarray,
    best: np.ndarray,
    *,
    observed_at_least_as_good: bool,
    delta_worse: float,
    delta_better: float = 0.0,
) -> np.ndarray:
    """Rotation angle for every Q-bit, following the table in the module docstring."""
    observed, best = np.asarray(observed), np.asarray(best)
    if observed_at_least_as_good:
        target, delta = observed, delta_better
    else:
        target, delta = best, delta_worse
    if delta == 0:
        return np.zeros(alpha.shape)
    return np.where(observed != best, delta * direction_towards(alpha, beta, target), 0.0)


def schedule_factor(progress: float, *, decay: bool, final_ratio: float) -> float:
    """Multiplier for the rotation angles: 1 for a constant schedule, else 1 -> ``final_ratio`` linearly."""
    if not decay:
        return 1.0
    progress = min(max(progress, 0.0), 1.0)
    return 1.0 - (1.0 - final_ratio) * progress
