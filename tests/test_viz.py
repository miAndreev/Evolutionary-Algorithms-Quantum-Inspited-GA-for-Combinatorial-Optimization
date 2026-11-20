import pytest

from qiga.experiments.analysis import load_results
from qiga.experiments.runner import run_experiment
from qiga.experiments.schema import ExperimentSpec
from qiga_viz import plot_convergence
from qiga_viz.__main__ import main as viz_main
from qiga_viz.style import SERIES_COLORS
from qiga_viz.style import color_map

SPEC = {
    "name": "viz",
    "seeds": 3,
    "max_evaluations": 200,
    "problems": [{"type": "knapsack", "n": 12}, {"type": "tsp", "n": 6}, {"type": "tsp", "n": 8}],
    "algorithms": [
        {"algorithm": "classical_ga", "params": {"population_size": 10}},
        {"algorithm": "random_search", "params": {"population_size": 10}},
    ],
}


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    return load_results(run_experiment(ExperimentSpec.model_validate(SPEC), tmp_path_factory.mktemp("viz")))


def test_color_follows_entity_order():
    assert color_map(["b", "a"]) == {"b": SERIES_COLORS[0], "a": SERIES_COLORS[1]}
    with pytest.raises(ValueError):
        color_map([str(i) for i in range(len(SERIES_COLORS) + 1)])


@pytest.mark.parametrize("metric", ["gap", "fitness"])
def test_one_panel_per_problem(results, metric):
    figure = plot_convergence(results, metric=metric)

    visible = [ax for ax in figure.axes if ax.get_visible()]
    assert len(visible) == 3
    assert [ax.get_title(loc="left") for ax in visible] == results.problems
    assert all(ax.get_xlim() == (0, 200) for ax in visible)


def test_series_colors_and_direct_labels(results):
    figure = plot_convergence(results, problems=results.problems[:1], band=False)

    ax = figure.axes[0]
    assert {line.get_color() for line in ax.get_lines()} == set(SERIES_COLORS[:2])
    assert sorted(text.get_text() for text in ax.texts) == ["classical_ga", "random_search"]
    assert [text.get_text() for text in figure.legends[0].get_texts()] == ["classical_ga", "random_search"]


def test_direct_labels_never_overlap(results):
    figure = plot_convergence(results, problems=results.problems[:1])

    positions = sorted(text.xy[1] for text in figure.axes[0].texts)
    assert positions[1] - positions[0] >= 0.06 - 1e-9


def test_zoom_ignores_early_outliers(results):
    zoomed = plot_convergence(results, problems=results.problems[:1], metric="gap").axes[0].get_ylim()
    full = plot_convergence(results, problems=results.problems[:1], metric="gap", zoom_after=None).axes[0].get_ylim()

    assert zoomed[1] - zoomed[0] <= full[1] - full[0]


def test_report_cli_writes_figures(results, tmp_path, capsys):
    assert viz_main(["report", str(results.folder), "-o", str(tmp_path)]) == 0

    assert capsys.readouterr().out.strip() == str(tmp_path)
    expected = {f"convergence_{m}.{s}" for m in ("gap", "fitness") for s in ("png", "pdf")} | {"summary.csv"}
    assert {p.name for p in tmp_path.iterdir()} == expected
