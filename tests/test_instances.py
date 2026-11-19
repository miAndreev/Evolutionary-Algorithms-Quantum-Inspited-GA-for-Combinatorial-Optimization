import pytest

from qiga.problems import KnapsackKind
from qiga.problems import generate_euclidean_tsp
from qiga.problems import generate_knapsack


@pytest.mark.parametrize("kind", list(KnapsackKind))
def test_knapsack_generator_is_reproducible(kind):
    assert generate_knapsack(50, kind, seed=1) == generate_knapsack(50, kind, seed=1)
    assert generate_knapsack(50, kind, seed=1) != generate_knapsack(50, kind, seed=2)


def test_knapsack_capacity_ratio():
    problem = generate_knapsack(100, capacity_ratio=0.3, seed=0)

    assert problem.capacity == int(0.3 * sum(problem.weights))


def test_strongly_correlated_values():
    problem = generate_knapsack(20, KnapsackKind.STRONGLY_CORRELATED, max_weight=100, seed=0)

    assert all(v == w + 10 for w, v in zip(problem.weights, problem.values, strict=True))


def test_weakly_correlated_values_within_spread():
    problem = generate_knapsack(200, KnapsackKind.WEAKLY_CORRELATED, max_weight=100, seed=0)

    assert all(v >= 1 and abs(v - w) <= 10 for w, v in zip(problem.weights, problem.values, strict=True))


def test_knapsack_generator_accepts_string_kind():
    assert generate_knapsack(5, "strongly_correlated", seed=0).name == "knapsack-strongly_correlated-n5-s0"


@pytest.mark.parametrize("kwargs", [{"n": 0}, {"n": 5, "capacity_ratio": 0}, {"n": 5, "capacity_ratio": 1.5}, {"n": 5, "kind": "bogus"}])
def test_knapsack_generator_rejects_bad_arguments(kwargs):
    with pytest.raises(ValueError):
        generate_knapsack(**kwargs)


def test_tsp_generator():
    problem = generate_euclidean_tsp(30, size=100.0, seed=4)

    assert problem.dimension == 30
    assert all(0 <= x <= 100 and 0 <= y <= 100 for x, y in problem.coordinates)
    assert problem == generate_euclidean_tsp(30, size=100.0, seed=4)


def test_tsp_generator_rejects_single_city():
    with pytest.raises(ValueError):
        generate_euclidean_tsp(1)
