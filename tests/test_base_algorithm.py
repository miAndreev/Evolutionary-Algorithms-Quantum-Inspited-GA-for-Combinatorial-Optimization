import pytest

from qiga.algorithms import EvolutionaryAlgorithm
from qiga.algorithms import RandomSearch
from qiga.algorithms import RunResult
from qiga.algorithms import StopReason
from qiga.config import GAConfig
from qiga.problems import generate_euclidean_tsp
from qiga.problems import generate_knapsack


@pytest.fixture
def knapsack():
    return generate_knapsack(20, seed=0)


@pytest.fixture
def tsp():
    return generate_euclidean_tsp(8, seed=0)


def test_stops_after_generations(knapsack):
    result = RandomSearch(knapsack, GAConfig(population_size=10, generations=5, seed=1)).run()

    assert result.stop_reason is StopReason.GENERATIONS
    assert result.generations == 5
    assert result.evaluations == 10 * 6  # initial population + 5 generations
    assert [r.generation for r in result.history] == list(range(6))
    assert [r.evaluations for r in result.history] == [10 * (g + 1) for g in range(6)]


def test_budget_cuts_generation_short(knapsack):
    result = RandomSearch(knapsack, GAConfig(population_size=10, generations=100, max_evaluations=35, seed=1)).run()

    assert result.stop_reason is StopReason.EVALUATIONS
    assert result.evaluations == 35
    assert result.history[-1].evaluations == 35
    assert result.history[-1].generation == 3


def test_budget_on_generation_boundary(knapsack):
    result = RandomSearch(knapsack, GAConfig(population_size=10, generations=100, max_evaluations=30, seed=1)).run()

    assert result.stop_reason is StopReason.EVALUATIONS
    assert result.evaluations == 30
    assert len(result.history) == 3


def test_stops_at_target(knapsack):
    target = knapsack.exact_optimum() * 0.5
    result = RandomSearch(knapsack, GAConfig(population_size=10, generations=1000, target_fitness=target, seed=1)).run()

    assert result.stop_reason is StopReason.TARGET
    assert result.best_fitness >= target
    assert result.evaluations_to_target is not None
    assert result.evaluations_to_target <= result.evaluations


def test_best_so_far_is_monotonic_and_consistent(tsp):
    result = RandomSearch(tsp, GAConfig(population_size=10, generations=30, seed=2)).run()

    best_so_far = [r.best_so_far for r in result.history]
    assert all(b <= a for a, b in zip(best_so_far, best_so_far[1:], strict=False))
    assert result.best_fitness == best_so_far[-1] == result.improvements[-1].fitness
    assert result.best_fitness == tsp.evaluate(result.best_solution)
    assert all(r.best <= r.mean <= r.worst for r in result.history)


def test_run_is_reproducible_and_rerunnable(knapsack):
    algorithm = RandomSearch(knapsack, GAConfig(population_size=10, generations=5, seed=3))

    first = algorithm.run()
    second = algorithm.run()

    assert first.best_solution == second.best_solution
    assert first.history == second.history


def test_different_seeds_differ(knapsack):
    first = RandomSearch(knapsack, GAConfig(population_size=10, generations=5, seed=1)).run()
    second = RandomSearch(knapsack, GAConfig(population_size=10, generations=5, seed=2)).run()

    assert first.history != second.history


def test_result_metadata_and_serialization(knapsack):
    config = GAConfig(population_size=10, generations=3, seed=4)
    result = RandomSearch(knapsack, config).run()

    assert result.algorithm == "random_search"
    assert result.problem == knapsack.name
    assert GAConfig.model_validate(result.config) == config
    assert result.elapsed_seconds >= 0
    assert RunResult.model_validate_json(result.model_dump_json()) == result


def test_result_frames(knapsack):
    result = RandomSearch(knapsack, GAConfig(population_size=10, generations=3, seed=4)).run()

    history = result.history_frame()
    convergence = result.convergence_frame()

    assert list(history.columns) == ["generation", "evaluations", "best_so_far", "best", "mean", "std", "worst"]
    assert len(history) == 4
    assert list(convergence.columns) == ["evaluations", "fitness"]
    assert convergence["fitness"].is_monotonic_increasing


class GreedyInit(EvolutionaryAlgorithm[GAConfig]):
    """Evaluates more than one population during initialization."""

    name = "greedy_init"

    def _initialize(self):
        for _ in range(self.config.population_size * 2):
            self.evaluate(self.problem.random_solution(self.rng))

    def _step(self):
        raise NotImplementedError

    def _population_fitness(self):
        return [0.0]


def test_budget_exhausted_during_initialization_raises(knapsack):
    with pytest.raises(ValueError, match="initial population"):
        GreedyInit(knapsack, GAConfig(population_size=10, max_evaluations=15)).run()


def test_config_rejects_budget_below_population():
    with pytest.raises(ValueError, match="max_evaluations"):
        GAConfig(population_size=10, max_evaluations=5)
