"""Fixed-target convergence speed: empirical CDF of the evaluations needed to reach target gaps.

For every run and every target gap t, record the first evaluation at which the best-so-far
gap was <= t. The curve at x is the fraction of (run, target) pairs reached within x
evaluations - higher and further left is faster. Pairs never reached keep the curve below 1.
This is the "runtime ECDF" used by benchmarking platforms such as COCO/IOHprofiler.
"""

from typing import TYPE_CHECKING

import numpy as np

from qiga.experiments.analysis import run_metrics
from qiga.experiments.analysis import target_column
from qiga_viz.style import LINE_WIDTH
from qiga_viz.style import TEXT_SECONDARY
from qiga_viz.style import color_map
from qiga_viz.style import label_line_ends
from qiga_viz.style import make_panels
from qiga_viz.style import series_legend
from qiga_viz.style import style_axes

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.figure import Figure
    import pandas as pd

    from qiga.experiments.analysis import Results

DEFAULT_TARGETS = (10.0, 5.0, 2.0, 1.0, 0.5)


def ecdf_curve(evaluations: Sequence[float], budget: float, grid: np.ndarray) -> np.ndarray:
    """Fraction of entries <= each grid value; ``NaN`` (never reached) entries count as misses."""
    values = np.asarray(evaluations, dtype=np.float64)
    if values.size == 0:
        return np.zeros(grid.shape)
    reached = np.sort(values[np.isfinite(values) & (values <= budget)])
    return np.searchsorted(reached, grid, side="right") / values.size


def plot_target_ecdf(
    results: Results,
    problems: list[str] | None = None,
    *,
    targets: Sequence[float] = DEFAULT_TARGETS,
    aggregate: bool = True,
    columns: int = 2,
    grid_points: int = 300,
) -> Figure:
    """One panel per problem (and, if ``aggregate`` and there are several problems, one over all of them)."""
    problems = problems or results.problems
    colors = color_map(results.algorithms)
    runs = run_metrics(results, targets=tuple(targets))
    target_columns = [target_column(target) for target in targets]
    panels: list[tuple[str, pd.DataFrame]] = [(problem, runs[runs["problem"] == problem]) for problem in problems]
    if aggregate and len(problems) > 1:
        panels.append((f"all {len(problems)} problems", runs[runs["problem"].isin(problems)]))
    figure, axes = make_panels(len(panels), columns)

    for ax, (title, subset) in zip(axes, panels, strict=True):
        budget = float(subset["evaluations"].max())
        grid = np.linspace(0, budget, grid_points)
        ends = []
        for name, color in colors.items():
            per_algorithm = subset[subset["algorithm"] == name]
            if per_algorithm.empty:
                continue
            curve = ecdf_curve(per_algorithm[target_columns].to_numpy().ravel(), budget, grid)
            ax.step(grid, curve, where="post", color=color, linewidth=LINE_WIDTH)
            ends.append((float(curve[-1]), name))
        style_axes(ax)
        ax.set_xlim(0, budget)
        ax.set_ylim(-0.02, 1.02)
        label_line_ends(ax, ends, colors)
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_xlabel("evaluations")
        ax.set_ylabel("fraction of (run, target) pairs reached")

    series_legend(figure, colors)
    shown = ", ".join(f"{target:g}%" for target in targets)
    figure.suptitle(f"Runtime ECDF - target gaps {shown}", x=0.99, ha="right", fontsize=9, color=TEXT_SECONDARY)
    return figure
