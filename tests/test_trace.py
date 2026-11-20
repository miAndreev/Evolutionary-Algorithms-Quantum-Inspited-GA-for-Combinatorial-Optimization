import numpy as np
import pytest

from qiga.algorithms import QIGA
from qiga.algorithms import ClassicalGA
from qiga.algorithms import run_traced
from qiga.config import ClassicalGAConfig
from qiga.config import QIGAConfig
from qiga.problems import KnapsackKind
from qiga.problems import generate_euclidean_tsp
from qiga.problems import generate_knapsack


@pytest.fixture
def tsp():
    return generate_euclidean_tsp(10, seed=0)


def test_trace_does_not_change_the_run(tsp):
    config = QIGAConfig(generations=10**9, max_evaluations=505, seed=3)

    traced, trace = run_traced(QIGA(tsp, config), every=7)
    plain = QIGA(tsp, config).run()

    assert traced.history == plain.history
    assert trace.frames[-1].evaluations == 505
    assert trace.frames[-1].best_fitness == plain.best_fitness


def test_frames_every_n_generations(tsp):
    _, trace = run_traced(QIGA(tsp, QIGAConfig(generations=20, seed=0)), every=5)

    assert [frame.generation for frame in trace.frames] == [0, 5, 10, 15, 20]
    assert all(frame.p_one.shape == (10 * 5,) for frame in trace.frames)
    assert np.allclose(trace.frames[0].p_one, 0.5)


def test_improvements_only_keep_changes(tsp):
    _, trace = run_traced(QIGA(tsp, QIGAConfig(generations=50, seed=0)))
    improvements = trace.improvements()

    fitness = [frame.best_fitness for frame in improvements]
    assert improvements[0] is trace.frames[0]
    assert all(later < earlier for earlier, later in zip(fitness, fitness[1:], strict=False))
    assert fitness[-1] == trace.frames[-1].best_fitness


def test_classical_ga_frames_have_no_qbits():
    problem = generate_knapsack(10, KnapsackKind.UNCORRELATED)

    _, trace = run_traced(ClassicalGA(problem, ClassicalGAConfig(population_size=10, generations=3, seed=0)))

    assert len(trace.frames) == 4
    assert all(frame.p_one is None for frame in trace.frames)


def test_every_must_be_positive(tsp):
    with pytest.raises(ValueError, match="positive"):
        run_traced(QIGA(tsp, QIGAConfig()), every=0)
