from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qiga.__main__ import main as qiga_main
from qiga.experiments.analysis import Results
from qiga.experiments.schema import ExperimentSpec
from qiga.experiments.stats import effect_magnitude
from qiga.experiments.stats import friedman_ranks
from qiga.experiments.stats import holm
from qiga.experiments.stats import metric_values
from qiga.experiments.stats import pairwise_tests
from qiga.experiments.stats import vargha_delaney_a12

ALGORITHMS = ["good", "bad", "same"]


def make_results(final: dict[str, dict[str, list[float]]]) -> Results:
    """Synthetic minimization results: ``final[problem][algorithm]`` lists each seed's best fitness.

    Every run improves once at evaluation 1 (fitness 100) and reaches its final value at 50.
    """
    problems = list(final)
    spec = ExperimentSpec.model_validate(
        {
            "name": "synthetic",
            "problems": [{"type": "tsp", "n": 5}],
            "algorithms": [{"algorithm": "random_search", "label": name} for name in ALGORITHMS],
        },
    )
    runs, convergence = [], []
    for problem in problems:
        for algorithm, values in final[problem].items():
            for seed, value in enumerate(values):
                key = {"problem": problem, "algorithm": algorithm, "seed": seed}
                runs.append({**key, "sense": "minimize", "reference": 10.0, "best_fitness": value, "evaluations": 100, "elapsed_seconds": 0.0})
                convergence.extend([{**key, "evaluations": 1, "fitness": 100.0}, {**key, "evaluations": 50, "fitness": value}])
    return Results(Path(), spec, pd.DataFrame(runs), pd.DataFrame(convergence), pd.DataFrame())


@pytest.fixture
def results():
    rng = np.random.default_rng(0)
    return make_results(
        {
            problem: {"good": list(10 + rng.random(15)), "bad": list(20 + rng.random(15)), "same": list(10 + rng.random(15))}
            for problem in ("p1", "p2", "p3")
        },
    )


def test_a12_extremes_and_ties():
    assert vargha_delaney_a12([1, 2], [3, 4]) == 1.0
    assert vargha_delaney_a12([3, 4], [1, 2]) == 0.0
    assert vargha_delaney_a12([1, 1], [1, 1]) == 0.5
    with pytest.raises(ValueError, match="at least one"):
        vargha_delaney_a12([], [1])


@pytest.mark.parametrize(("a12", "label"), [(0.5, "negligible"), (0.6, "small"), (0.32, "medium"), (0.9, "large"), (0.0, "large")])
def test_effect_magnitude(a12, label):
    assert effect_magnitude(a12) == label


def test_holm_adjustment():
    np.testing.assert_allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    np.testing.assert_allclose(holm([0.5, 0.9]), [1.0, 1.0])


def test_pairwise_finds_the_better_algorithm(results):
    table = pairwise_tests(results)

    assert len(table) == 3 * 3  # 3 pairs on 3 problems
    good_vs_bad = table[(table["a"] == "good") & (table["b"] == "bad")]
    assert (good_vs_bad["winner"] == "good").all()
    assert (good_vs_bad["a12"] == 1.0).all()
    assert (good_vs_bad["p_holm"] >= good_vs_bad["p_value"]).all()
    assert (table[(table["a"] == "good") & (table["b"] == "same")]["winner"] == "-").all()


def test_identical_samples_are_not_significant():
    results = make_results({"p": {"good": [12.0] * 5, "bad": [12.0] * 5}})

    row = pairwise_tests(results).iloc[0]

    assert row["p_value"] == 1.0
    assert row["winner"] == "-"


def test_metric_values(results):
    gap = metric_values(results, "gap")
    anytime = metric_values(results, "anytime")
    target = metric_values(results, "target", target=50.0)

    assert (anytime["value"] > gap["value"]).all()  # the curve starts far above the final gap
    assert set(target["value"]) == {50.0, np.inf}  # good/same reach a 50% gap at evaluation 50, bad never does


def test_friedman_ranks(results):
    ranks, p_value = friedman_ranks(results)

    assert ranks["bad"] == 3.0
    assert p_value is not None
    assert p_value < 0.1


def test_friedman_needs_two_problems_and_three_algorithms():
    single = make_results({"p": {"good": [11.0, 12.0], "bad": [20.0, 21.0], "same": [11.0, 12.0]}})
    tied = make_results({p: {name: [11.0, 12.0] for name in ALGORITHMS} for p in ("a", "b")})

    assert friedman_ranks(single)[1] is None
    assert friedman_ranks(tied)[1] == 1.0


@pytest.mark.parametrize("metric", ["gap", "anytime", "target"])
def test_stats_cli(tmp_path, capsys, metric):
    spec = {
        "name": "cli",
        "seeds": 3,
        "max_evaluations": 100,
        "problems": [{"type": "knapsack", "n": 10}, {"type": "tsp", "n": 6}],
        "algorithms": [
            {"algorithm": "classical_ga", "params": {"population_size": 10}},
            {"algorithm": "random_search", "params": {"population_size": 10}},
        ],
    }
    spec_path = tmp_path / "spec.yaml"
    ExperimentSpec.model_validate(spec).to_yaml(spec_path)
    qiga_main(["run", str(spec_path), "-o", str(tmp_path)])
    folder = capsys.readouterr().out.strip()

    assert qiga_main(["stats", folder, "-m", metric, "-t", "5"]) == 0

    out = capsys.readouterr().out
    assert "p_holm" in out
    assert "Friedman p = n/a" in out
