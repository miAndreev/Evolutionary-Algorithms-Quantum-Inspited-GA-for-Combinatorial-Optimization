"""Convergence plot: best-so-far fitness (or gap to reference) against evaluations."""

from typing import TYPE_CHECKING
from typing import Literal

from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import numpy as np

from qiga.experiments.analysis import Results
from qiga.experiments.analysis import best_so_far_on_grid
from qiga.experiments.analysis import gap_percent
from qiga.experiments.analysis import reference_values
from qiga_viz.style import BAND_ALPHA
from qiga_viz.style import LINE_WIDTH
from qiga_viz.style import SURFACE
from qiga_viz.style import TEXT_PRIMARY
from qiga_viz.style import TEXT_SECONDARY
from qiga_viz.style import color_map
from qiga_viz.style import style_axes

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    import pandas as pd

type Metric = Literal["fitness", "gap"]

_LABEL_MIN_SPACING = 0.06  # in axes-height fractions


def plot_convergence(
    results: Results,
    problems: list[str] | None = None,
    *,
    metric: Metric = "gap",
    band: bool = True,
    grid_points: int = 200,
    columns: int = 2,
    zoom_after: float | None = 0.02,
) -> Figure:
    """One panel per problem: median over seeds (line) and interquartile range (band).

    ``metric="gap"`` plots the distance to the reference in percent (lower is better for
    every problem); ``metric="fitness"`` plots the raw objective.
    ``zoom_after`` fits the y-axis to the data after that fraction of the budget, so the
    first random (often infeasible) solutions do not flatten the rest of the curve;
    ``None`` shows everything.
    """
    problems = problems or results.problems
    colors = color_map(results.algorithms)
    references = reference_values(results.runs)
    columns = min(columns, len(problems))
    rows = -(-len(problems) // columns)

    figure = Figure(figsize=(5.2 * columns, 3.6 * rows + 0.6), facecolor=SURFACE, layout="constrained")
    axes = figure.subplots(rows, columns, squeeze=False).ravel()
    for ax in axes[len(problems) :]:
        ax.set_visible(False)

    for ax, problem in zip(axes, problems, strict=False):
        budget = int(results.runs.loc[results.runs["problem"] == problem, "evaluations"].max())
        grid = np.unique(np.linspace(1, budget, grid_points).astype(int))
        samples = best_so_far_on_grid(results, problem, grid)
        if metric == "gap":
            samples["fitness"] = gap_percent(samples["fitness"], references[problem])
        _draw_panel(ax, samples, colors, band=band, zoom_from=None if zoom_after is None else zoom_after * budget)
        style_axes(ax)
        ax.set_title(problem, fontsize=10, loc="left")
        ax.set_xlabel("evaluations")
        ax.set_ylabel("gap to reference (%)" if metric == "gap" else "best-so-far fitness")
        ax.set_xlim(0, budget)

    handles = [Line2D([], [], color=colors[name], linewidth=LINE_WIDTH, label=name) for name in results.algorithms]
    figure.legend(handles=handles, loc="outside upper left", ncols=len(handles), frameon=False, labelcolor=TEXT_PRIMARY, fontsize=9)
    return figure


def _draw_panel(ax: Axes, samples: pd.DataFrame, colors: dict[str, str], *, band: bool, zoom_from: float | None) -> None:
    ends: list[tuple[float, str]] = []
    visible: list[np.ndarray] = []
    for algorithm, color in colors.items():
        data = samples[samples["algorithm"] == algorithm]
        if data.empty:
            continue
        by_evaluations = data.groupby("evaluations")["fitness"]
        lower, median, upper = (by_evaluations.quantile(q).to_numpy() for q in (0.25, 0.5, 0.75))
        x = by_evaluations.quantile(0.5).index.to_numpy()
        if band:
            ax.fill_between(x, lower, upper, color=color, alpha=BAND_ALPHA, linewidth=0, step="post")
        ax.step(x, median, where="post", color=color, linewidth=LINE_WIDTH)
        ends.append((float(median[-1]), algorithm))
        if zoom_from is not None:
            shown = x >= zoom_from
            visible.extend([lower[shown], upper[shown]] if band else [median[shown]])
    ax.margins(x=0)
    values = np.concatenate(visible) if visible else np.array([])
    values = values[np.isfinite(values)]
    if values.size:
        low, high = values.min(), values.max()
        pad = (high - low) * 0.08 or abs(high) * 0.05 or 1.0
        ax.set_ylim(low - pad, high + pad)
    _label_line_ends(ax, ends, colors)


def _label_line_ends(ax: Axes, ends: list[tuple[float, str]], colors: dict[str, str]) -> None:
    """Direct labels at the right edge, nudged apart so they never overlap."""
    if not ends:
        return
    low, high = ax.get_ylim()  # get_ylim applies pending autoscaling
    span = (high - low) or 1.0
    positions = sorted(((value - low) / span, name) for value, name in ends)
    placed: list[float] = []
    for y, _ in positions:
        placed.append(max(y, placed[-1] + _LABEL_MIN_SPACING) if placed else y)
    overflow = placed[-1] - 1.0
    if overflow > 0:
        placed = [y - overflow for y in placed]
    for y, (_, name) in zip(placed, positions, strict=True):
        ax.annotate(
            name,
            xy=(1.0, y),
            xycoords="axes fraction",
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=TEXT_SECONDARY,
            annotation_clip=False,
        )
        ax.plot([1.0], [y], transform=ax.transAxes, marker="o", markersize=4, color=colors[name], clip_on=False)
