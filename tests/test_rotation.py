import numpy as np
import pytest

from qiga.operators.rotation import direction_towards
from qiga.operators.rotation import lookup_table_angles
from qiga.operators.rotation import schedule_factor
from qiga.representations import gates

S = gates.INV_SQRT2
# One Q-bit per case of the sign table: alpha*beta > 0, alpha*beta < 0, alpha = 0, beta = 0.
ALPHA = np.array([S, S, 0.0, 1.0])
BETA = np.array([S, -S, 1.0, 0.0])


def test_direction_towards_one_matches_han_kim_row_3():
    np.testing.assert_array_equal(direction_towards(ALPHA, BETA, np.ones(4)), [1, -1, 0, 1])


def test_direction_towards_zero_matches_han_kim_row_5():
    np.testing.assert_array_equal(direction_towards(ALPHA, BETA, np.zeros(4)), [-1, 1, 1, 0])


@pytest.mark.parametrize("target", [0, 1])
def test_rotating_by_direction_moves_probability_towards_target(target):
    rng = np.random.default_rng(0)
    alpha, beta = gates.normalize(rng.normal(size=200), rng.normal(size=200))
    targets = np.full(200, target)

    delta = 0.01
    _, new_beta = gates.rotate(alpha, beta, delta * direction_towards(alpha, beta, targets))

    # Q-bits closer than delta to the target pole overshoot it (as in Han & Kim's table); skip those.
    distance_to_target = np.arcsin(np.abs(alpha if target == 1 else beta))
    far = distance_to_target > delta
    moved = (new_beta**2 - beta**2)[far]
    assert far.sum() > 150
    assert (moved > 0).all() if target == 1 else (moved < 0).all()


def test_lookup_rotates_differing_bits_towards_best_when_observation_is_worse():
    alpha, beta = np.full(4, S), np.full(4, S)
    observed, best = np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1])

    angles = lookup_table_angles(alpha, beta, observed, best, observed_at_least_as_good=False, delta_worse=0.1)

    np.testing.assert_allclose(angles, [0.0, 0.1, -0.1, 0.0])


def test_lookup_rotates_towards_observation_when_it_is_better():
    alpha, beta = np.full(4, S), np.full(4, S)
    observed, best = np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1])

    paper_default = lookup_table_angles(alpha, beta, observed, best, observed_at_least_as_good=True, delta_worse=0.1)
    with_theta4 = lookup_table_angles(alpha, beta, observed, best, observed_at_least_as_good=True, delta_worse=0.1, delta_better=0.05)

    np.testing.assert_array_equal(paper_default, 0.0)
    np.testing.assert_allclose(with_theta4, [0.0, -0.05, 0.05, 0.0])


def test_repeated_updates_converge_to_attractor():
    rng = np.random.default_rng(1)
    best = rng.integers(0, 2, size=30)
    alpha, beta = np.full(30, S), np.full(30, S)

    for _ in range(300):
        observed = gates.measure(beta, rng)
        angles = lookup_table_angles(alpha, beta, observed, best, observed_at_least_as_good=False, delta_worse=0.01 * np.pi)
        alpha, beta = gates.rotate(alpha, beta, angles)

    np.testing.assert_array_equal(np.round(beta**2), best)


@pytest.mark.parametrize(
    ("progress", "decay", "expected"),
    [(0.5, False, 1.0), (0.0, True, 1.0), (0.5, True, 0.55), (1.0, True, 0.1), (2.0, True, 0.1), (-1.0, True, 1.0)],
)
def test_schedule_factor(progress, decay, expected):
    assert schedule_factor(progress, decay=decay, final_ratio=0.1) == pytest.approx(expected)
