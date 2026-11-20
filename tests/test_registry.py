import random

import pytest

from qiga.operators.registry import make_crossover
from qiga.operators.registry import make_mutation
from qiga.operators.registry import make_selection
from qiga.operators.registry import mut_inversion
from qiga.problems import Encoding
from qiga.problems import Sense


@pytest.mark.parametrize("name", [None, "one_point", "two_point", "uniform"])
def test_binary_crossovers_keep_bits(name):
    random.seed(0)
    first, second = [0] * 10, [1] * 10

    make_crossover(name, Encoding.BINARY)(first, second)

    assert len(first) == len(second) == 10
    assert sorted(first + second) == [0] * 10 + [1] * 10


@pytest.mark.parametrize("name", [None, "ordered", "pmx"])
def test_permutation_crossovers_keep_permutations(name):
    random.seed(0)
    for _ in range(50):
        first, second = random.sample(range(12), 12), random.sample(range(12), 12)
        make_crossover(name, Encoding.PERMUTATION)(first, second)
        assert sorted(first) == sorted(second) == list(range(12))


@pytest.mark.parametrize("name", [None, "shuffle", "inversion"])
def test_permutation_mutations_keep_permutations(name):
    random.seed(0)
    mutate = make_mutation(name, Encoding.PERMUTATION, gene_prob=0.3)
    for _ in range(50):
        individual = random.sample(range(12), 12)
        mutate(individual)
        assert sorted(individual) == list(range(12))


def test_flip_bit_rate():
    random.seed(0)
    mutate = make_mutation("flip_bit", Encoding.BINARY, gene_prob=1.0)
    individual = [0, 1, 0, 1]

    mutate(individual)

    assert individual == [1, 0, 1, 0]


def test_inversion_reverses_a_segment():
    random.seed(1)
    individual = list(range(10))

    mut_inversion(individual)

    changed = [i for i, gene in enumerate(individual) if gene != i]
    start, end = changed[0], changed[-1]
    assert individual[start : end + 1] == list(range(end, start - 1, -1))


@pytest.mark.parametrize(
    ("factory", "name", "encoding"),
    [
        (make_crossover, "ordered", Encoding.BINARY),
        (make_crossover, "two_point", Encoding.PERMUTATION),
        (make_mutation, "flip_bit", Encoding.PERMUTATION),
        (make_mutation, "inversion", Encoding.BINARY),
    ],
)
def test_operator_incompatible_with_encoding(factory, name, encoding):
    args = (name, encoding) if factory is make_crossover else (name, encoding, 0.1)
    with pytest.raises(ValueError, match="not available"):
        factory(*args)


def test_roulette_rejects_minimization():
    with pytest.raises(ValueError, match="roulette"):
        make_selection("roulette", Sense.MINIMIZE, 3)


def test_unknown_selection():
    with pytest.raises(ValueError, match="unknown selection"):
        make_selection("bogus", Sense.MAXIMIZE, 3)
