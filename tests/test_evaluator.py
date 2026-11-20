import pytest

from qiga.algorithms import BudgetExhaustedError
from qiga.algorithms import Evaluator
from qiga.problems import TSP
from qiga.problems import Knapsack


@pytest.fixture
def knapsack():
    return Knapsack(weights=[10, 20, 30], values=[60, 100, 120], capacity=50)


@pytest.fixture
def tsp():
    return TSP.from_coordinates([(0, 0), (0, 3), (4, 3), (4, 0)])


def test_counts_and_tracks_best_for_maximization(knapsack):
    evaluator = Evaluator(knapsack)

    evaluator([1, 0, 0])  # 60
    evaluator([0, 1, 1])  # 220
    evaluator([1, 1, 0])  # 160, not an improvement

    assert evaluator.evaluations == 3
    assert evaluator.best_fitness == 220
    assert evaluator.best_solution == [0, 1, 1]
    assert evaluator.improvements == [(1, 60), (2, 220)]


def test_tracks_best_for_minimization(tsp):
    evaluator = Evaluator(tsp)

    evaluator([0, 2, 1, 3])  # 18
    evaluator([0, 1, 2, 3])  # 14

    assert evaluator.best_fitness == pytest.approx(14)
    assert [e for e, _ in evaluator.improvements] == [1, 2]


def test_best_solution_is_a_copy(knapsack):
    evaluator = Evaluator(knapsack)
    solution = [1, 0, 0]

    evaluator(solution)
    solution[0] = 0

    assert evaluator.best_solution == [1, 0, 0]


def test_budget_is_enforced(knapsack):
    evaluator = Evaluator(knapsack, max_evaluations=2)

    evaluator([1, 0, 0])
    assert evaluator.remaining == 1
    evaluator([0, 1, 0])

    assert evaluator.budget_exhausted
    with pytest.raises(BudgetExhaustedError):
        evaluator([0, 0, 1])
    assert evaluator.evaluations == 2


def test_unlimited_budget(knapsack):
    evaluator = Evaluator(knapsack)

    assert evaluator.remaining is None
    assert not evaluator.budget_exhausted


@pytest.mark.parametrize(("target", "hit_at"), [(160, 2), (220, 3), (221, None)])
def test_target_hit_maximization(knapsack, target, hit_at):
    evaluator = Evaluator(knapsack, target_fitness=target)

    for solution in ([1, 0, 0], [1, 1, 0], [0, 1, 1]):
        evaluator(solution)

    assert evaluator.target_hit_at == hit_at
    assert evaluator.target_reached == (hit_at is not None)


def test_target_hit_minimization(tsp):
    evaluator = Evaluator(tsp, target_fitness=15)

    evaluator([0, 2, 1, 3])
    assert not evaluator.target_reached
    evaluator([0, 1, 2, 3])

    assert evaluator.target_hit_at == 2
