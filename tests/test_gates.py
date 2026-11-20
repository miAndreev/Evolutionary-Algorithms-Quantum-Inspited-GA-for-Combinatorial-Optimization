import numpy as np
import pytest

from qiga.representations import gates


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def random_amplitudes(rng, n=100):
    return gates.normalize(rng.normal(size=n), rng.normal(size=n))


def test_normalize_scales_to_unit_length():
    alpha, beta = gates.normalize([3, 0, -1], [4, 2, 1])

    np.testing.assert_allclose(alpha**2 + beta**2, 1.0)
    np.testing.assert_allclose(alpha, [0.6, 0.0, -np.sqrt(0.5)])


@pytest.mark.parametrize(("alpha", "beta"), [(0, 0), (np.nan, 1), (np.inf, 1), ([1, 0], [0, 0])])
def test_normalize_rejects_invalid(alpha, beta):
    with pytest.raises(ValueError):
        gates.normalize(alpha, beta)


@pytest.mark.parametrize(
    ("theta", "expected"),
    [
        (np.pi / 2, (0.0, 1.0)),
        (np.pi, (-1.0, 0.0)),
        (-np.pi / 2, (0.0, -1.0)),
        (np.pi / 4, (gates.INV_SQRT2, gates.INV_SQRT2)),
    ],
)
def test_rotation_of_zero_state(theta, expected):
    alpha, beta = gates.rotate(np.array(1.0), np.array(0.0), theta)

    assert (alpha, beta) == pytest.approx(expected)


def test_rotation_preserves_norm_and_inverts(rng):
    alpha, beta = random_amplitudes(rng)
    theta = rng.uniform(-np.pi, np.pi, size=alpha.shape)

    rotated = gates.rotate(alpha, beta, theta)
    restored = gates.rotate(*rotated, -theta)

    np.testing.assert_allclose(rotated[0] ** 2 + rotated[1] ** 2, 1.0)
    np.testing.assert_allclose(restored, (alpha, beta), atol=1e-12)


def test_positive_rotation_in_first_quadrant_increases_p_one():
    alpha, beta = gates.normalize(np.array([0.9, 0.5]), np.array([0.1, 0.5]))

    _, rotated_beta = gates.rotate(alpha, beta, 0.05)

    assert (rotated_beta**2 > beta**2).all()


def test_quantum_not_swaps_amplitudes():
    alpha, beta = np.array([0.6, 0.8]), np.array([0.8, -0.6])

    assert [a.tolist() for a in gates.quantum_not(alpha, beta)] == [[0.8, -0.6], [0.6, 0.8]]
    assert [a.tolist() for a in gates.quantum_not(alpha, beta, [True, False])] == [[0.8, 0.8], [0.6, -0.6]]


def test_quantum_not_does_not_alias_inputs():
    alpha, beta = np.array([0.6]), np.array([0.8])

    new_alpha, _ = gates.quantum_not(alpha, beta)
    new_alpha[0] = 0

    assert beta[0] == 0.8


def test_h_epsilon_clamps_only_collapsed_qbits():
    alpha = np.array([1.0, 0.0, -1.0, 0.6, 0.0])
    beta = np.array([0.0, 1.0, 0.0, 0.8, -1.0])

    new_alpha, new_beta = gates.h_epsilon(alpha, beta, 0.01)

    np.testing.assert_allclose(new_beta**2, [0.01, 0.99, 0.01, 0.64, 0.99])
    np.testing.assert_allclose(new_alpha**2 + new_beta**2, 1.0)
    assert new_alpha[3] == 0.6 and new_beta[3] == 0.8, "Q-bits inside the range are untouched"
    assert new_alpha[2] < 0 and new_beta[4] < 0, "signs are preserved"


def test_h_epsilon_zero_is_identity(rng):
    alpha, beta = random_amplitudes(rng)

    np.testing.assert_array_equal(gates.h_epsilon(alpha, beta, 0.0), (alpha, beta))


@pytest.mark.parametrize("epsilon", [-0.1, 0.5, 1.0])
def test_h_epsilon_rejects_bad_epsilon(epsilon):
    with pytest.raises(ValueError):
        gates.h_epsilon(np.array([1.0]), np.array([0.0]), epsilon)


def test_measure_deterministic_states(rng):
    beta = np.array([0.0, 1.0, -1.0, 0.0])

    np.testing.assert_array_equal(gates.measure(beta, rng), [0, 1, 1, 0])


def test_measure_frequencies_match_probabilities(rng):
    p_one = np.array([0.1, 0.3, 0.5, 0.9])
    beta = np.tile(np.sqrt(p_one), (100_000, 1))

    frequencies = gates.measure(beta, rng).mean(axis=0)

    np.testing.assert_allclose(frequencies, p_one, atol=0.01)
