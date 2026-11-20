import numpy as np
import pytest

from qiga.algorithms import QIGA
from qiga.algorithms import RandomSearch
from qiga.algorithms import StopReason
from qiga.config import GAConfig
from qiga.config import QIGAConfig
from qiga.experiments.runner import run_experiment
from qiga.experiments.analysis import load_results
from qiga.experiments.schema import ExperimentSpec
from qiga.problems import KnapsackKind
from qiga.problems import generate_euclidean_tsp
from qiga.problems import generate_knapsack


@pytest.fixture
def knapsack():
    return generate_knapsack(30, KnapsackKind.WEAKLY_CORRELATED, seed=0)


def run(problem, **kwargs):
    algorithm = QIGA(problem, QIGAConfig(**{"generations": 50, "seed": 1, **kwargs}))
    return algorithm, algorithm.run()


def test_rejects_permutation_problems():
    with pytest.raises(ValueError, match="binary"):
        QIGA(generate_euclidean_tsp(5), QIGAConfig())


def test_result_is_valid(knapsack):
    _, result = run(knapsack)

    assert result.algorithm == "qiga"
    assert set(result.best_solution) <= {0, 1}
    assert result.best_fitness == knapsack.evaluate(result.best_solution)


def test_one_evaluation_per_individual_per_generation(knapsack):
    _, result = run(knapsack, population_size=10, generations=20)

    assert [r.evaluations for r in result.history] == [10 * (g + 1) for g in range(21)]


def test_reaches_optimum_with_repair(knapsack):
    _, result = run(knapsack, generations=300, repair=True)

    assert result.best_fitness == knapsack.exact_optimum()


def test_repair_makes_observations_feasible(knapsack):
    algorithm, _ = run(knapsack, generations=5, repair=True)

    assert all(knapsack.is_feasible(solution) for solution in algorithm.observed)


def test_qbits_collapse_towards_global_best(knapsack):
    algorithm, _ = run(knapsack, generations=500)

    p_one = np.mean([q.p_one() for q in algorithm.qpopulation], axis=0)
    agreement = np.mean(np.round(p_one) == np.array(algorithm.attractors[0]))
    assert agreement > 0.9
    assert np.abs(p_one - 0.5).mean() > 0.3, "probabilities should have moved away from 1/2"


def test_attractors_are_never_worse_than_observations(knapsack):
    algorithm, result = run(knapsack, generations=30)

    for attractor, fitness in zip(algorithm.attractors, algorithm.attractor_fitness, strict=True):
        assert knapsack.evaluate(attractor) == fitness
        assert fitness <= result.best_fitness
    assert all(a >= o for a, o in zip(algorithm.attractor_fitness, algorithm.observed_fitness, strict=True))


def test_global_migration_shares_the_best(knapsack):
    algorithm, result = run(knapsack, generations=10, global_migration_period=10, local_group_size=1)

    assert all(fitness == result.best_fitness for fitness in algorithm.attractor_fitness)
    assert all(attractor == algorithm.attractors[0] for attractor in algorithm.attractors)


def test_local_migration_shares_within_groups(knapsack):
    algorithm, _ = run(knapsack, population_size=6, generations=7, local_group_size=3, global_migration_period=None)

    groups = [algorithm.attractor_fitness[:3], algorithm.attractor_fitness[3:]]
    assert all(len(set(group)) == 1 for group in groups)


def test_no_migration_keeps_attractors_independent(knapsack):
    algorithm, _ = run(knapsack, population_size=10, generations=3, local_group_size=1, global_migration_period=None)

    assert len({tuple(attractor) for attractor in algorithm.attractors}) > 1


def test_h_epsilon_keeps_qbits_open(knapsack):
    algorithm, _ = run(knapsack, generations=300, epsilon=0.02)

    p_one = np.concatenate([q.p_one() for q in algorithm.qpopulation])
    assert p_one.min() >= 0.02 - 1e-12
    assert p_one.max() <= 0.98 + 1e-12


def test_not_gate_with_certainty_flips_every_qbit(knapsack):
    algorithm = QIGA(knapsack, QIGAConfig(generations=1, not_gate_prob=1.0, delta_theta=1e-9, delta_theta_min=1e-9, seed=1))
    algorithm.run()

    np.testing.assert_allclose(np.concatenate([q.p_one() for q in algorithm.qpopulation]), 0.5, atol=1e-6)
    biased = QIGA(knapsack, QIGAConfig(generations=1, not_gate_prob=1.0, seed=1))
    biased._initialize()
    biased.qpopulation = [type(q).from_probabilities(np.full(knapsack.dimension, 0.9)) for q in biased.qpopulation]
    biased._apply_gates()
    np.testing.assert_allclose(biased.qpopulation[0].p_one(), 0.1)


def test_linear_decay_runs_with_budget(knapsack):
    algorithm = QIGA(knapsack, QIGAConfig(generations=10**9, max_evaluations=300, rotation_schedule="linear_decay", seed=1))

    result = algorithm.run()

    assert result.evaluations == 300
    assert algorithm._progress() == 1.0


def test_progress_by_generations(knapsack):
    algorithm, _ = run(knapsack, generations=20)

    assert algorithm._progress() == 1.0


def test_budget_cut_mid_generation_keeps_state_consistent(knapsack):
    algorithm = QIGA(knapsack, QIGAConfig(population_size=10, generations=10**9, max_evaluations=55, seed=1))

    result = algorithm.run()

    assert result.stop_reason is StopReason.EVALUATIONS
    assert result.evaluations == 55
    assert len(algorithm.observed) == len(algorithm.observed_fitness) == 10


def test_run_is_reproducible(knapsack):
    _, first = run(knapsack, generations=20)
    _, second = run(knapsack, generations=20)

    assert first.history == second.history
    assert first.best_solution == second.best_solution


def test_beats_random_search_at_equal_budget():
    problem = generate_knapsack(100, KnapsackKind.UNCORRELATED, seed=3)
    budget = {"generations": 10**9, "max_evaluations": 3000}
    qiga = [QIGA(problem, QIGAConfig(**budget, seed=s)).run().best_fitness for s in range(5)]
    rs = [RandomSearch(problem, GAConfig(**budget, population_size=10, seed=s)).run().best_fitness for s in range(5)]

    assert min(qiga) > max(rs)


def test_runs_inside_an_experiment(tmp_path):
    spec = ExperimentSpec.model_validate(
        {
            "name": "qiga",
            "seeds": 2,
            "max_evaluations": 200,
            "problems": [{"type": "knapsack", "n": 20}],
            "algorithms": [{"algorithm": "qiga", "params": {"epsilon": 0.01}}, {"algorithm": "classical_ga"}],
        },
    )

    results = load_results(run_experiment(spec, tmp_path))

    assert results.algorithms == ["qiga", "classical_ga"]
    assert (results.runs["evaluations"] == 200).all()
