import numpy as np
import pytest

from qiga.representations import QBit
from qiga.representations import QChromosome

# --- QBit -----------------------------------------------------------------------


def test_qbit_default_is_equal_superposition():
    assert QBit().probabilities() == pytest.approx((0.5, 0.5))


def test_qbit_normalizes():
    qbit = QBit(alpha=3, beta=4)

    assert (qbit.alpha, qbit.beta) == pytest.approx((0.6, 0.8))


@pytest.mark.parametrize(("alpha", "beta"), [(0, 0), (float("nan"), 1)])
def test_qbit_rejects_invalid(alpha, beta):
    with pytest.raises(ValueError):
        QBit(alpha, beta)


def test_qbit_rotation():
    qbit = QBit(alpha=1, beta=0)

    qbit.rotate(np.pi / 2)

    assert (qbit.alpha, qbit.beta) == pytest.approx((0.0, 1.0))


def test_qbit_not_and_h_epsilon():
    qbit = QBit(alpha=1, beta=0)

    qbit.apply_not()
    assert qbit.probabilities() == pytest.approx((0.0, 1.0))

    qbit.apply_h_epsilon(0.05)
    assert qbit.probabilities() == pytest.approx((0.05, 0.95))


def test_qbit_measurement():
    rng = np.random.default_rng(42)

    assert QBit(1, 0).measure(rng) == 0
    assert QBit(0, 1).measure(rng) == 1
    samples = [QBit().measure(np.random.default_rng(7)) for _ in range(3)]
    assert len(set(samples)) == 1, "same seed, same observation"


def test_qbit_repr():
    assert repr(QBit(1, 0)) == "QBit(alpha=1.0000, beta=0.0000)"


# --- QChromosome -------------------------------------------------------------------


def test_chromosome_default_superposition():
    chromosome = QChromosome(5)

    assert len(chromosome) == chromosome.dimension == 5
    np.testing.assert_allclose(chromosome.probabilities(), 0.5)
    assert chromosome.amplitudes.shape == (5, 2)


def test_chromosome_from_amplitudes_normalizes_rows():
    chromosome = QChromosome(amplitudes=[[3, 4], [0, 2]])

    np.testing.assert_allclose(chromosome.amplitudes, [[0.6, 0.8], [0.0, 1.0]])


def test_chromosome_from_probabilities():
    chromosome = QChromosome.from_probabilities([0.0, 0.25, 1.0])

    np.testing.assert_allclose(chromosome.p_one(), [0.0, 0.25, 1.0])


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"dimension": 0},
        {"amplitudes": [[0, 0], [1, 0]]},
        {"amplitudes": [1, 0]},
        {"amplitudes": np.empty((0, 2))},
        {"dimension": 3, "amplitudes": [[1, 0]]},
    ],
)
def test_chromosome_rejects_invalid(kwargs):
    with pytest.raises(ValueError):
        QChromosome(**kwargs)


@pytest.mark.parametrize("p_one", [[-0.1], [1.5], [[0.5]]])
def test_from_probabilities_rejects_invalid(p_one):
    with pytest.raises(ValueError):
        QChromosome.from_probabilities(p_one)


def test_amplitude_views_are_read_only():
    chromosome = QChromosome(3)

    with pytest.raises(ValueError):
        chromosome.beta[0] = 1.0
    chromosome.amplitudes[0, 0] = 99.0  # a copy: no effect

    assert chromosome.alpha[0] == pytest.approx(0.70710678)


def test_measurement_is_binary_reproducible_and_non_destructive():
    chromosome = QChromosome(50)
    before = chromosome.amplitudes

    first = chromosome.measure(np.random.default_rng(3))
    second = chromosome.measure(np.random.default_rng(3))

    assert set(first.tolist()) <= {0, 1}
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(chromosome.amplitudes, before)


def test_scalar_and_per_gene_rotation():
    chromosome = QChromosome(amplitudes=[[1, 0], [1, 0], [1, 0]])

    chromosome.rotate(np.pi / 2)
    np.testing.assert_allclose(chromosome.p_one(), 1.0)

    chromosome.rotate([0.0, -np.pi / 2, -np.pi / 4])
    np.testing.assert_allclose(chromosome.p_one(), [1.0, 0.0, 0.5], atol=1e-12)


@pytest.mark.parametrize("theta", [[0.1, 0.2], [[0.1, 0.2, 0.3]]])
def test_rotation_rejects_wrong_shape(theta):
    with pytest.raises(ValueError):
        QChromosome(3).rotate(theta)


def test_many_small_rotations_stay_normalized():
    chromosome = QChromosome(10)
    rng = np.random.default_rng(0)

    for _ in range(10_000):
        chromosome.rotate(rng.uniform(-0.05, 0.05, size=10))

    np.testing.assert_allclose(chromosome.probabilities().sum(axis=1), 1.0, atol=1e-12)


def test_not_with_mask_and_h_epsilon():
    chromosome = QChromosome.from_probabilities([0.0, 0.0, 1.0])

    chromosome.apply_not([True, False, False])
    np.testing.assert_allclose(chromosome.p_one(), [1.0, 0.0, 1.0])

    chromosome.apply_not()
    np.testing.assert_allclose(chromosome.p_one(), [0.0, 1.0, 0.0])

    chromosome.apply_h_epsilon(0.02)
    np.testing.assert_allclose(chromosome.p_one(), [0.02, 0.98, 0.02])


def test_not_rejects_wrong_mask():
    with pytest.raises(ValueError):
        QChromosome(3).apply_not([True])


def test_copy_is_independent():
    chromosome = QChromosome(5)

    clone = chromosome.copy()
    clone.rotate(0.5)

    assert not np.allclose(chromosome.amplitudes, clone.amplitudes)
    assert type(clone) is QChromosome


def test_indexing_returns_qbit_copy():
    chromosome = QChromosome(amplitudes=[[0.6, 0.8], [1, 0]])

    qbit = chromosome[0]
    qbit.rotate(1.0)

    assert (qbit.alpha, qbit.beta) != pytest.approx((0.6, 0.8))
    assert chromosome[0].probabilities() == pytest.approx((0.36, 0.64))


def test_repr():
    assert repr(QChromosome(4)) == "QChromosome(dimension=4, mean_p_one=0.500)"
