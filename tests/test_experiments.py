import numpy as np
import pandas as pd
import pytest

from qiga.__main__ import main as qiga_main
from qiga.config import ClassicalGAConfig
from qiga.experiments.analysis import best_so_far_on_grid
from qiga.experiments.analysis import gap_percent
from qiga.experiments.analysis import load_results
from qiga.experiments.analysis import reference_values
from qiga.experiments.analysis import summary_table
from qiga.experiments.runner import run_experiment
from qiga.experiments.schema import CONVERGENCE_COLUMNS
from qiga.experiments.schema import HISTORY_COLUMNS
from qiga.experiments.schema import RUN_COLUMNS
from qiga.experiments.schema import AlgorithmSpec
from qiga.experiments.schema import ExperimentSpec

SPEC = {
    "name": "unit",
    "seeds": 3,
    "max_evaluations": 300,
    "problems": [{"type": "knapsack", "n": 15, "seed": 1}, {"type": "tsp", "n": 7, "seed": 1}],
    "algorithms": [
        {"algorithm": "classical_ga", "params": {"population_size": 20}},
        {"algorithm": "random_search", "params": {"population_size": 20}},
    ],
}


@pytest.fixture(scope="module")
def results_folder(tmp_path_factory):
    return run_experiment(ExperimentSpec.model_validate(SPEC), tmp_path_factory.mktemp("results"))


@pytest.fixture(scope="module")
def results(results_folder):
    return load_results(results_folder)


# --- schema -----------------------------------------------------------------


def test_spec_yaml_round_trip(tmp_path):
    spec = ExperimentSpec.model_validate(SPEC)

    spec.to_yaml(tmp_path / "spec.yaml")

    assert ExperimentSpec.from_yaml(tmp_path / "spec.yaml") == spec


def test_seeds_as_count_or_list():
    assert ExperimentSpec.model_validate(SPEC).seed_list == [0, 1, 2]
    assert ExperimentSpec.model_validate({**SPEC, "seeds": [7, 11]}).seed_list == [7, 11]


def test_problem_specs_build_problems():
    spec = ExperimentSpec.model_validate(SPEC)

    knapsack, tsp = (problem.build() for problem in spec.problems)

    assert knapsack.dimension == 15
    assert tsp.dimension == 7


def test_experiment_budget_overrides_generations():
    config = AlgorithmSpec(algorithm="classical_ga", params={"population_size": 10}).config_for(seed=4, max_evaluations=500)

    assert isinstance(config, ClassicalGAConfig)
    assert config.seed == 4
    assert config.max_evaluations == 500
    assert config.generations > 10**6


def test_explicit_generations_are_kept_with_budget():
    config = AlgorithmSpec(algorithm="classical_ga", params={"generations": 7}).config_for(seed=0, max_evaluations=500)

    assert config.generations == 7


@pytest.mark.parametrize(
    "algorithm",
    [
        {"algorithm": "bogus"},
        {"algorithm": "classical_ga", "params": {"mutation_prob": 2}},
        {"algorithm": "classical_ga", "params": {"seed": 1}},
        {"algorithm": "random_search", "params": {"crossover": "pmx"}},
    ],
)
def test_invalid_algorithm_spec(algorithm):
    with pytest.raises(ValueError):
        ExperimentSpec.model_validate({**SPEC, "algorithms": [algorithm]})


def test_duplicate_algorithm_names_need_labels():
    duplicated = [{"algorithm": "classical_ga"}, {"algorithm": "classical_ga", "params": {"crossover": "uniform"}}]
    with pytest.raises(ValueError, match="unique"):
        ExperimentSpec.model_validate({**SPEC, "algorithms": duplicated})

    labelled = [duplicated[0], {**duplicated[1], "label": "ga_uniform"}]
    assert [a.name for a in ExperimentSpec.model_validate({**SPEC, "algorithms": labelled}).algorithms] == ["classical_ga", "ga_uniform"]


@pytest.mark.parametrize("name", ["", "a/b", "../up"])
def test_experiment_name_is_a_safe_folder_name(name):
    with pytest.raises(ValueError):
        ExperimentSpec.model_validate({**SPEC, "name": name})


# --- runner -----------------------------------------------------------------


def test_results_folder_layout(results_folder):
    assert results_folder.parent.name == "unit"
    assert sorted(p.name for p in results_folder.iterdir()) == ["convergence.parquet", "experiment.yaml", "history.parquet", "runs.parquet"]


def test_tables_have_expected_shape(results):
    assert tuple(results.runs.columns) == RUN_COLUMNS
    assert tuple(results.convergence.columns) == CONVERGENCE_COLUMNS
    assert tuple(results.history.columns) == HISTORY_COLUMNS
    assert len(results.runs) == 2 * 2 * 3
    assert (results.runs["evaluations"] == 300).all()
    assert (results.runs["stop_reason"] == "evaluations").all()


def test_reference_is_known_optimum_only_for_knapsack(results):
    by_problem = results.runs.groupby("problem")["reference"].first()

    assert by_problem.filter(like="knapsack").notna().all()
    assert by_problem.filter(like="tsp").isna().all()


def test_convergence_matches_final_result(results):
    last = results.convergence.groupby(["problem", "algorithm", "seed"])["fitness"].last()
    final = results.runs.set_index(["problem", "algorithm", "seed"])["best_fitness"]

    pd.testing.assert_series_equal(last.sort_index(), final.sort_index(), check_names=False)


def test_parallel_matches_sequential(tmp_path, results):
    parallel = load_results(run_experiment(ExperimentSpec.model_validate(SPEC), tmp_path, workers=2))

    columns = ["problem", "algorithm", "seed", "best_fitness", "evaluations"]
    pd.testing.assert_frame_equal(parallel.runs[columns], results.runs[columns])
    pd.testing.assert_frame_equal(parallel.convergence, results.convergence)


def test_progress_callback(tmp_path):
    calls = []

    run_experiment(ExperimentSpec.model_validate({**SPEC, "seeds": 1}), tmp_path, progress=lambda done, total: calls.append((done, total)))

    assert calls == [(1, 4), (2, 4), (3, 4), (4, 4)]


# --- loading & analysis -------------------------------------------------------


def test_load_latest_run_from_experiment_folder(results_folder):
    assert load_results(results_folder.parent).folder.parent == results_folder.parent


def test_load_missing_results(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_results(tmp_path)
    (tmp_path / "runs.parquet").touch()
    with pytest.raises(FileNotFoundError, match="missing"):
        load_results(tmp_path)


def test_algorithm_and_problem_order(results):
    assert results.algorithms == ["classical_ga", "random_search"]
    assert results.problems == ["knapsack-uncorrelated-n15-s1", "tsp-euclid-n7-s1"]


def test_reference_values_fallback_to_best_found():
    runs = pd.DataFrame(
        {
            "problem": ["a", "a", "b", "b"],
            "sense": ["minimize", "minimize", "maximize", "maximize"],
            "reference": [None, None, 50.0, 50.0],
            "best_fitness": [12.0, 10.0, 40.0, 45.0],
        },
    )

    assert reference_values(runs).to_dict() == {"a": 10.0, "b": 50.0}


def test_gap_percent():
    np.testing.assert_allclose(gap_percent(np.array([90, 100, 110]), 100), [10, 0, 10])
    np.testing.assert_allclose(gap_percent(np.array([0, 2]), 0), [0, 200])


def test_best_so_far_on_grid_is_a_step_function(results):
    problem = results.problems[0]
    grid = np.array([0, 1, 50, 300, 1000])

    samples = best_so_far_on_grid(results, problem, grid)

    assert len(samples) == 2 * 3 * len(grid)
    first_run = samples[(samples["algorithm"] == "classical_ga") & (samples["seed"] == 0)]["fitness"].to_numpy()
    assert np.isnan(first_run[0])
    assert np.all(np.diff(first_run[1:]) >= 0)
    final = results.runs.query("problem == @problem and algorithm == 'classical_ga' and seed == 0")["best_fitness"].iloc[0]
    assert first_run[-1] == first_run[-2] == final


def test_summary_table(results):
    table = summary_table(results)

    assert len(table) == 4
    assert (table["runs"] == 3).all()
    assert (table["gap_mean_percent"] >= 0).all()
    assert table["hit_rate"].between(0, 1).all()


# --- CLI ----------------------------------------------------------------------


def test_cli_run_and_summary(tmp_path, capsys):
    spec_path = tmp_path / "spec.yaml"
    ExperimentSpec.model_validate({**SPEC, "seeds": 1}).to_yaml(spec_path)

    assert qiga_main(["run", str(spec_path), "-o", str(tmp_path / "out")]) == 0
    folder = capsys.readouterr().out.strip()
    assert qiga_main(["summary", folder]) == 0

    assert "classical_ga" in capsys.readouterr().out
