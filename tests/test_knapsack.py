import itertools

import numpy as np
import pytest

from qiga.problems import Encoding
from qiga.problems import Knapsack
from qiga.problems import KnapsackKind
from qiga.problems import Sense
from qiga.problems import generate_knapsack


@pytest.fixture
def small():
    return Knapsack(weights=[10, 20, 30], values=[60, 100, 120], capacity=50)


def brute_force_optimum(problem):
    return max(problem.evaluate(bits) for bits in itertools.product((0, 1), repeat=problem.dimension))


def test_metadata(small):
    assert small.sense is Sense.MAXIMIZE
    assert small.encoding is Encoding.BINARY
    assert small.dimension == 3


def test_feasible_solution(small):
    solution = [1, 1, 0]

    assert small.evaluate(solution) == 160
    assert small.total_weight(solution) == 30
    assert small.total_value(solution) == 160
    assert small.is_feasible(solution)


def test_infeasible_solution_gets_zero(small):
    solution = [1, 1, 1]

    assert not small.is_feasible(solution)
    assert small.evaluate(solution) == 0


def test_accepts_numpy_and_bool_arrays(small):
    assert small.evaluate(np.array([1, 1, 0], dtype=np.int8)) == 160
    assert small.evaluate(np.array([True, True, False])) == 160


@pytest.mark.parametrize("solution", [[1, 0], [1, 0, 0, 1], [1, 2, 0], [0.5, 1, 0], [[1, 0, 1]]])
def test_malformed_solution_raises(small, solution):
    with pytest.raises(ValueError):
        small.evaluate(solution)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"weights": [10, 20], "values": [1], "capacity": 5},
        {"weights": [], "values": [], "capacity": 5},
        {"weights": [0, 20], "values": [1, 2], "capacity": 5},
        {"weights": [10, 20], "values": [-1, 2], "capacity": 5},
        {"weights": [10, 20], "values": [1, 2], "capacity": -1},
    ],
)
def test_invalid_instance_raises(kwargs):
    with pytest.raises(ValueError):
        Knapsack(**kwargs)


def test_instance_is_immutable(small):
    with pytest.raises(ValueError):
        small.capacity = 1000


def test_random_solution_is_reproducible(small):
    first = small.random_solution(np.random.default_rng(1))
    second = small.random_solution(np.random.default_rng(1))

    assert first == second
    assert set(first) <= {0, 1}
    assert len(first) == small.dimension


@pytest.mark.parametrize("seed", range(10))
def test_repair_returns_feasible_and_maximal_solution(seed):
    problem = generate_knapsack(30, KnapsackKind.WEAKLY_CORRELATED, seed=seed)
    rng = np.random.default_rng(seed)

    repaired = problem.repair(rng.integers(0, 2, size=problem.dimension))

    assert problem.is_feasible(repaired)
    free = problem.capacity - problem.total_weight(repaired)
    unused = [w for w, bit in zip(problem.weights, repaired, strict=True) if not bit]
    assert all(w > free for w in unused), "repair should add every item that still fits"


def test_repair_keeps_feasible_solution_value_or_improves(small):
    assert small.evaluate(small.repair([1, 0, 0])) >= small.evaluate([1, 0, 0])


@pytest.mark.parametrize("kind", list(KnapsackKind))
@pytest.mark.parametrize("seed", range(5))
def test_exact_optimum_matches_brute_force(kind, seed):
    problem = generate_knapsack(12, kind, max_weight=50, seed=seed)

    assert problem.exact_optimum() == brute_force_optimum(problem)


def test_exact_optimum_when_no_item_fits():
    problem = Knapsack(weights=[10, 20], values=[5, 6], capacity=5)

    assert problem.exact_optimum() == 0


def test_is_better_respects_sense(small):
    assert small.is_better(20, 10)
    assert not small.is_better(10, 10)


def test_equal_instances_compare_equal_after_use(small):
    other = Knapsack(weights=[10, 20, 30], values=[60, 100, 120], capacity=50)
    small.evaluate([1, 0, 0])  # populates cached arrays on one instance only

    assert small == other
