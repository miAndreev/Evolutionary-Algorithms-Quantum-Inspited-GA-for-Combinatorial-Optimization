import itertools

import numpy as np
import pytest

from qiga.problems import TSP
from qiga.problems import Encoding
from qiga.problems import Sense
from qiga.problems import generate_euclidean_tsp

DISTANCES = [
    [0, 10, 15],
    [10, 0, 20],
    [15, 20, 0],
]


@pytest.fixture
def triangle():
    return TSP(distances=DISTANCES)


def test_metadata(triangle):
    assert triangle.sense is Sense.MINIMIZE
    assert triangle.encoding is Encoding.PERMUTATION
    assert triangle.dimension == 3


def test_closed_tour_length(triangle):
    assert triangle.evaluate([0, 1, 2]) == 45
    assert triangle.evaluate(np.array([2, 0, 1])) == 45


def test_asymmetric_tour_uses_direction():
    problem = TSP(distances=[[0, 1, 9], [9, 0, 1], [1, 9, 0]])

    assert problem.evaluate([0, 1, 2]) == 3
    assert problem.evaluate([0, 2, 1]) == 27


@pytest.mark.parametrize("route", [[0, 1, 1], [0, 1, 3], [-1, 0, 1]])
def test_non_permutation_is_infeasible_and_raises(triangle, route):
    assert not triangle.is_feasible(route)
    with pytest.raises(ValueError):
        triangle.evaluate(route)


@pytest.mark.parametrize("route", [[0, 1], [0, 1, 2, 3], [0.0, 1.0, 2.0]])
def test_malformed_route_raises(triangle, route):
    with pytest.raises(ValueError):
        triangle.is_feasible(route)


@pytest.mark.parametrize(
    "distances",
    [
        [[0, 10], [10]],
        [[0]],
        [[1, 10], [10, 0]],
        [[0, -1], [1, 0]],
    ],
)
def test_invalid_matrix_raises(distances):
    with pytest.raises(ValueError):
        TSP(distances=distances)


def test_from_coordinates_square():
    problem = TSP.from_coordinates([(0, 0), (0, 3), (4, 3), (4, 0)])

    assert problem.evaluate([0, 1, 2, 3]) == pytest.approx(14.0)
    assert problem.evaluate([0, 2, 1, 3]) == pytest.approx(5 + 4 + 5 + 4)
    assert problem.coordinates is not None


def test_from_coordinates_rounding():
    problem = TSP.from_coordinates([(0, 0), (1, 1)], rounded=True)

    assert problem.distances[0][1] == 1.0


def test_coordinates_length_must_match():
    with pytest.raises(ValueError):
        TSP(distances=DISTANCES, coordinates=[(0, 0)])


def test_random_solution_is_permutation_and_reproducible():
    problem = generate_euclidean_tsp(20, seed=3)

    first = problem.random_solution(np.random.default_rng(5))

    assert sorted(first) == list(range(20))
    assert first == problem.random_solution(np.random.default_rng(5))


def test_generator_matches_brute_force_sanity():
    problem = generate_euclidean_tsp(6, seed=0)
    best = min(problem.evaluate((0, *rest)) for rest in itertools.permutations(range(1, 6)))

    assert best <= problem.evaluate(list(range(6)))
    assert best > 0


def test_from_coordinates_rejects_wrong_shape():
    with pytest.raises(ValueError):
        TSP.from_coordinates([(0, 0, 0), (1, 1, 1)])


def test_is_better_respects_sense(triangle):
    assert triangle.is_better(10, 20)
    assert not triangle.is_better(20, 10)


def test_repair_is_identity_for_tsp(triangle):
    assert triangle.repair(np.array([2, 0, 1])) == [2, 0, 1]
