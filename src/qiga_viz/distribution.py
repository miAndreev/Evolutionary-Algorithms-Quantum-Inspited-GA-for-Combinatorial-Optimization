"""Final-quality distributions: one box per algorithm, every run shown as a dot."""

from typing import TYPE_CHECKING
from typing import Literal

import numpy as np

from qiga.experiments.analysis import run_metrics
from qiga_viz.style import TEXT_SECONDARY
from qiga_viz.style import colors_or_neutral
from qiga_viz.style import make_panels
from qiga_viz.style import style_axes

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from qiga.experiments.analysis import Results

type DistributionMetric = Literal["gap", "anytime"]

_AXIS_LABELS = {"gap": "final gap to reference (%)", "anytime": "anytime gap: mean gap over the budget (%)"}


def plot_final_distribution(
    results: Results,
    problems: list[str] | None = None,
    *,
    metric: DistributionMetric = "gap",
    columns: int = 2,
    seed: int = 0,
) -> Figure:
    """One panel per problem, algorithms top to bottom in experiment order (lower is better).

    Boxes are horizontal so long variant names (e.g. from sweeps) stay readable; the
    jitter of the dots is seeded so the figure is reproducible.
    """
    problems = problems or results.problems
    runs = run_metrics(results)
    column = "gap_percent" if metric == "gap" else "anytime_gap"
    names = results.algorithms
    colors = colors_or_neutral(names)
    rng = np.random.default_rng(seed)
    figure, axes = make_panels(len(problems), columns, height=max(2.4, 0.45 * len(names) + 1.2))

    for ax, problem in zip(axes, problems, strict=True):
        subset = runs[runs["problem"] == problem]
        present = [name for name in names if (subset["algorithm"] == name).any()]
        positions = np.arange(len(present))[::-1]  # first algorithm on top
        samples = [subset.loc[subset["algorithm"] == name, column].to_numpy() for name in present]
        boxes = ax.boxplot(
            samples,
            positions=positions,
            orientation="horizontal",
            widths=0.55,
            showfliers=False,
            patch_artist=True,
            medianprops={"color": TEXT_SECONDARY, "linewidth": 2},
            whiskerprops={"color": TEXT_SECONDARY, "linewidth": 1},
            capprops={"color": TEXT_SECONDARY, "linewidth": 1},
            boxprops={"linewidth": 0},
        )
        for patch, name in zip(boxes["boxes"], present, strict=True):
            patch.set_facecolor(colors[name])
            patch.set_alpha(0.28)
        for position, sample, name in zip(positions, samples, present, strict=True):
            jitter = rng.uniform(-0.18, 0.18, sample.size)
            ax.scatter(sample, position + jitter, s=14, color=colors[name], edgecolors="white", linewidths=0.6, zorder=3)
        ax.set_yticks(positions, present)
        style_axes(ax)
        ax.grid(visible=False, axis="y")
        ax.set_title(problem, fontsize=10, loc="left")
        ax.set_xlabel(_AXIS_LABELS[metric])
    return figure
