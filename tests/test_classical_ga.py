import itertools

import numpy as np
import pytest

from qiga.algorithms import ClassicalGA
from qiga.algorithms import RandomSearch
from qiga.algorithms import StopReason
from qiga.config import ClassicalGAConfig
from qiga.config import GAConfig
from qiga.problems import TSP
from qiga.problems import Knapsack
from qiga.problems import KnapsackKind
from qiga.problems import generate_euclidean_tsp
from qiga.problems import generate_knapsack


@pytest.fixture
def knapsack():
    return generate_knapsack(30, KnapsackKind.WEAKLY_CORRELATED, seed=0)


@pytest.fixture
def tsp():
    return generate_euclidean_tsp(8, seed=0)


def tsp_optimum(problem):
    return min(problem.evaluate((0, *rest)) for rest in itertools.permutations(range(1, problem.dimension)))


def test_knapsack_result_is_valid(knapsack):
    result = ClassicalGA(knapsack, ClassicalGAConfig(population_size=30, generations=20, seed=1)).run()

    assert result.algorithm == "classical_ga"
    assert set(result.best_solution) <= {0, 1}
    assert result.best_fitness == knapsack.evaluate(result.best_solution)
    assert knapsack.is_feasible(result.best_solution)


def test_knapsack_with_repair_reaches_optimum(knapsack):
    config = ClassicalGAConfig(population_size=50, generations=150, repair=True, seed=1)

    result = ClassicalGA(knapsack, config).run()

    assert result.best_fitness == knapsack.exact_optimum()


def test_repair_makes_whole_population_feasible(knapsack):
    algorithm = ClassicalGA(knapsack, ClassicalGAConfig(population_size=30, generations=5, repair=True, seed=1))

    algorithm.run()

    assert all(knapsack.is_feasible(individual) for individual in algorithm.population)


@pytest.mark.parametrize("crossover", ["ordered", "pmx"])
@pytest.mark.parametrize("mutation", ["inversion", "shuffle"])
def test_tsp_finds_small_optimum(tsp, crossover, mutation):
    config = ClassicalGAConfig(population_size=50, generations=100, crossover=crossover, mutation=mutation, seed=1)

    result = ClassicalGA(tsp, config).run()

    assert sorted(result.best_solution) == list(range(tsp.dimension))
    assert result.best_fitness == pytest.approx(tsp_optimum(tsp))


def test_population_stays_valid_permutations():
    problem = generate_euclidean_tsp(15, seed=2)
    algorithm = ClassicalGA(problem, ClassicalGAConfig(population_size=20, generations=10, seed=1))

    algorithm.run()

    assert all(problem.is_feasible(individual) for individual in algorithm.population)


def test_every_changed_offspring_is_evaluated_once(knapsack):
    """Regression: a parent selected twice must become two independent offspring."""
    config = ClassicalGAConfig(population_size=20, generations=10, elitism=2, crossover_prob=1.0, mutation_prob=1.0, seed=1)

    result = ClassicalGA(knapsack, config).run()

    assert [r.evaluations for r in result.history] == [20 + g * 18 for g in range(11)]


def test_offspring_are_independent_objects(knapsack):
    algorithm = ClassicalGA(knapsack, ClassicalGAConfig(population_size=20, generations=3, seed=1))

    algorithm.run()

    assert len({id(individual) for individual in algorithm.population}) == 20
    assert len({id(individual.fitness) for individual in algorithm.population}) == 20


def test_unchanged_offspring_are_not_reevaluated(knapsack):
    config = ClassicalGAConfig(population_size=20, generations=10, crossover_prob=0.0, mutation_prob=0.0, seed=1)

    result = ClassicalGA(knapsack, config).run()

    assert result.evaluations == 20


@pytest.mark.parametrize("problem_fixture", ["knapsack", "tsp"])
def test_elitism_keeps_population_best_monotonic(request, problem_fixture):
    problem = request.getfixturevalue(problem_fixture)
    config = ClassicalGAConfig(population_size=20, generations=30, elitism=1, seed=3)

    result = ClassicalGA(problem, config).run()

    best = [r.best for r in result.history]
    assert all(not problem.is_better(previous, current) for previous, current in itertools.pairwise(best))
    assert result.history[-1].best == result.best_fitness


def test_run_is_reproducible(tsp):
    config = ClassicalGAConfig(population_size=20, generations=10, seed=5)

    first = ClassicalGA(tsp, config).run()
    second = ClassicalGA(tsp, config).run()

    assert first.history == second.history
    assert first.best_solution == second.best_solution


def test_roulette_selection_on_knapsack(knapsack):
    config = ClassicalGAConfig(population_size=20, generations=10, selection="roulette", repair=True, seed=1)

    result = ClassicalGA(knapsack, config).run()

    assert knapsack.is_feasible(result.best_solution)


def test_evaluation_budget_is_exact(knapsack):
    config = ClassicalGAConfig(population_size=20, generations=1000, max_evaluations=500, seed=1)

    result = ClassicalGA(knapsack, config).run()

    assert result.stop_reason is StopReason.EVALUATIONS
    assert result.evaluations == 500


def test_stops_at_target(knapsack):
    target = 0.9 * knapsack.exact_optimum()
    config = ClassicalGAConfig(population_size=20, generations=1000, target_fitness=target, repair=True, seed=1)

    result = ClassicalGA(knapsack, config).run()

    assert result.stop_reason is StopReason.TARGET
    assert result.best_fitness >= target


@pytest.mark.parametrize(
    ("problem", "config"),
    [
        (Knapsack(weights=[1, 2], values=[1, 2], capacity=2), ClassicalGAConfig(crossover="ordered")),
        (TSP(distances=[[0, 1], [1, 0]]), ClassicalGAConfig(mutation="flip_bit")),
        (TSP(distances=[[0, 1], [1, 0]]), ClassicalGAConfig(selection="roulette")),
    ],
)
def test_incompatible_configuration_rejected_at_construction(problem, config):
    with pytest.raises(ValueError):
        ClassicalGA(problem, config)


def test_beats_random_search_at_equal_budget():
    problem = generate_euclidean_tsp(25, seed=7)
    budget = 3000
    ga, rs = [], []
    for seed in range(5):
        ga.append(
            ClassicalGA(problem, ClassicalGAConfig(population_size=30, generations=10_000, max_evaluations=budget, seed=seed)).run().best_fitness
        )
        rs.append(RandomSearch(problem, GAConfig(population_size=30, generations=10_000, max_evaluations=budget, seed=seed)).run().best_fitness)

    assert np.mean(ga) < np.mean(rs)
    assert max(ga) < min(rs)
