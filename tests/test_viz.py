from pathlib import Path

import numpy as np
import pytest

from qiga.algorithms import QIGA
from qiga.algorithms import ClassicalGA
from qiga.algorithms import run_traced
from qiga.config import ClassicalGAConfig
from qiga.config import QIGAConfig
from qiga.experiments.analysis import load_results
from qiga.experiments.runner import run_experiment
from qiga.experiments.schema import ExperimentSpec
from qiga.problems import TSP
from qiga.problems import generate_euclidean_tsp
from qiga_viz import animate_tours
from qiga_viz import plot_convergence
from qiga_viz import plot_final_distribution
from qiga_viz import plot_qbit_heatmap
from qiga_viz import plot_sweep
from qiga_viz import plot_target_ecdf
from qiga_viz import plot_tour
from qiga_viz.__main__ import main as viz_main
from qiga_viz.ecdf import ecdf_curve
from qiga_viz.style import NEUTRAL
from qiga_viz.style import SERIES_COLORS
from qiga_viz.style import color_map
from qiga_viz.style import colors_or_neutral
from qiga_viz.sweep import sweep_series

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


SWEEP_SPEC = {
    "name": "viz_sweep",
    "seeds": 2,
    "max_evaluations": 100,
    "problems": [{"type": "knapsack", "n": 12}],
    "algorithms": [
        {"algorithm": "qiga", "sweep": {"delta_theta": ["0.01pi", "0.05pi"], "population_size": [5, 10]}},
        {"algorithm": "qiga", "label": "eps", "sweep": {"epsilon": [0.0, 0.01, 0.02, 0.03, 0.04]}},
        {"algorithm": "classical_ga", "params": {"population_size": 10}},
    ],
}
FIGURES = {
    f"{name}.{suffix}"
    for name in ("convergence_gap", "convergence_fitness", "ecdf_targets", "distribution_gap", "distribution_anytime")
    for suffix in ("png", "pdf")
}
TABLES = {"summary.csv"} | {f"stats_{kind}_{metric}.csv" for kind in ("pairwise", "ranks") for metric in ("gap", "anytime")}


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    return load_results(run_experiment(ExperimentSpec.model_validate(SPEC), tmp_path_factory.mktemp("viz")))


@pytest.fixture(scope="module")
def sweep_results(tmp_path_factory):
    return load_results(run_experiment(ExperimentSpec.model_validate(SWEEP_SPEC), tmp_path_factory.mktemp("sweep")))


@pytest.fixture(scope="module")
def tsp():
    return generate_euclidean_tsp(8, seed=0)


@pytest.fixture(scope="module")
def qiga_trace(tsp):
    return run_traced(QIGA(tsp, QIGAConfig(generations=40, seed=0, bits_per_key=3)), every=2)


def test_color_follows_entity_order():
    assert color_map(["b", "a"]) == {"b": SERIES_COLORS[0], "a": SERIES_COLORS[1]}
    with pytest.raises(ValueError, match="at most"):
        color_map([str(i) for i in range(len(SERIES_COLORS) + 1)])


def test_too_many_series_fall_back_to_neutral():
    many = [str(i) for i in range(len(SERIES_COLORS) + 1)]

    assert colors_or_neutral(["a"]) == {"a": SERIES_COLORS[0]}
    assert set(colors_or_neutral(many).values()) == {NEUTRAL}


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


# --- distributions, ECDF, sweeps -------------------------------------------------


@pytest.mark.parametrize("metric", ["gap", "anytime"])
def test_distribution_has_one_box_row_per_algorithm(results, metric):
    figure = plot_final_distribution(results, metric=metric)

    ax = figure.axes[0]
    assert [label.get_text() for label in ax.get_yticklabels()] == results.algorithms
    assert len(ax.collections) == len(results.algorithms)  # one dot cloud each
    assert sum(len(c.get_offsets()) for c in ax.collections) == 2 * 3  # every run is a dot


def test_ecdf_curve():
    grid = np.array([0, 5, 10, 20])

    np.testing.assert_allclose(ecdf_curve([5, 10, np.nan, 30], budget=20, grid=grid), [0, 0.25, 0.5, 0.5])
    np.testing.assert_allclose(ecdf_curve([], budget=20, grid=grid), 0)


def test_ecdf_panels_include_aggregate(results):
    figure = plot_target_ecdf(results, targets=(50.0, 5.0))

    titles = [ax.get_title(loc="left") for ax in figure.axes if ax.get_visible()]
    assert titles == [*results.problems, "all 3 problems"]
    for ax in figure.axes[:3]:
        assert all(0 <= y <= 1 for line in ax.get_lines()[:2] for y in line.get_ydata())


def test_ecdf_single_problem_has_no_aggregate(results):
    assert len(plot_target_ecdf(results, results.problems[:1]).axes) == 1


def test_sweep_series_group_by_other_parameters(sweep_results):
    series = sweep_series(sweep_results.spec.variants, "delta_theta")

    assert list(series) == ["qiga[population_size=5]", "qiga[population_size=10]"]
    assert [value for value, _ in series["qiga[population_size=5]"]] == ["0.01pi", "0.05pi"]
    assert list(sweep_series(sweep_results.spec.variants, "epsilon")) == ["eps"]


def test_sweep_plot(sweep_results):
    ax = plot_sweep(sweep_results, "epsilon", metric="anytime").axes[0]

    assert [label.get_text() for label in ax.get_xticklabels()] == ["0.0", "0.01", "0.02", "0.03", "0.04"]
    assert ax.get_xlabel() == "epsilon"
    with pytest.raises(ValueError, match="sweeps"):
        plot_sweep(sweep_results, "tournament_size")


def test_distribution_with_many_variants_uses_neutral_color(sweep_results):
    ax = plot_final_distribution(sweep_results).axes[0]

    assert len(ax.get_yticklabels()) == 10


# --- traced runs ------------------------------------------------------------------


def test_qbit_heatmap(qiga_trace):
    _, trace = qiga_trace

    figure = plot_qbit_heatmap(trace, group_size=3)

    image = figure.axes[0].get_images()[0]
    assert image.get_array().shape == (8 * 3, len(trace.frames))
    assert len(figure.axes[0].get_lines()) == 8 - 1  # separators between the cities' keys


def test_qbit_heatmap_needs_qiga():
    problem = generate_euclidean_tsp(6)
    _, trace = run_traced(ClassicalGA(problem, ClassicalGAConfig(population_size=10, generations=2, seed=0)))

    with pytest.raises(ValueError, match="QIGA"):
        plot_qbit_heatmap(trace)


def test_tour_plot_and_animation(tsp, qiga_trace, tmp_path):
    result, trace = qiga_trace

    figure = plot_tour(tsp, result.best_solution)
    assert f"{result.best_fitness:.1f}" in figure.axes[0].get_title(loc="left")

    animation = animate_tours(tsp, trace, label="qiga")
    animation.save(tmp_path / "tour.gif", writer="pillow", dpi=40)
    assert (tmp_path / "tour.gif").stat().st_size > 0


def test_tour_needs_coordinates():
    problem = TSP(distances=[[0, 1, 1], [1, 0, 1], [1, 1, 0]])

    with pytest.raises(ValueError, match="coordinates"):
        plot_tour(problem, [0, 1, 2])


def test_animation_needs_frames(tsp, qiga_trace):
    _, trace = qiga_trace

    with pytest.raises(ValueError, match="empty"):
        animate_tours(tsp, type(trace)())


# --- CLI ----------------------------------------------------------------------------


def test_report_cli_writes_figures(results, tmp_path, capsys):
    assert viz_main(["report", str(results.folder), "-o", str(tmp_path)]) == 0

    assert capsys.readouterr().out.strip() == str(tmp_path)
    assert {p.name for p in tmp_path.iterdir()} == FIGURES | TABLES


def test_report_with_sweeps_skips_line_plots_when_too_many_variants(sweep_results, tmp_path):
    viz_main(["report", str(sweep_results.folder), "-o", str(tmp_path)])

    sweeps = {f"sweep_{p}.{s}" for p in ("delta_theta", "population_size", "epsilon") for s in ("png", "pdf")}
    distributions = {f"distribution_{m}.{s}" for m in ("gap", "anytime") for s in ("png", "pdf")}
    assert {p.name for p in tmp_path.iterdir()} == sweeps | distributions | TABLES


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (["tsp", "-n", "6", "-e", "200"], {"qbits.png", "qbits.pdf", "tour.png", "tour.pdf", "tour.gif"}),
        (["knapsack", "-n", "10", "-e", "200"], {"qbits.png", "qbits.pdf"}),
        (["tsp", "-n", "6", "-e", "200", "-a", "classical_ga"], {"tour.png", "tour.pdf", "tour.gif"}),
    ],
)
def test_demo_cli(tmp_path, capsys, args, expected):
    assert viz_main(["demo", *args, "-o", str(tmp_path)]) == 0

    folder = capsys.readouterr().out.strip()
    assert {p.name for p in Path(folder).iterdir()} == expected
