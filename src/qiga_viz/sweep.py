"""Parameter sensitivity: final (or anytime) gap against the values of one swept parameter."""

from typing import TYPE_CHECKING

import numpy as np

from qiga.experiments.analysis import run_metrics
from qiga_viz.style import LINE_WIDTH
from qiga_viz.style import color_map
from qiga_viz.style import make_panels
from qiga_viz.style import series_legend
from qiga_viz.style import style_axes

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from qiga.experiments.analysis import Results
    from qiga.experiments.schema import Variant
    from qiga_viz.distribution import DistributionMetric


def sweep_series(variants: list[Variant], parameter: str) -> dict[str, list[tuple[object, str]]]:
    """Group the variants that sweep ``parameter`` into series of ``(value, variant name)``.

    A series is one family with all *other* swept parameters fixed, e.g. sweeping
    ``delta_theta`` x ``bits_per_key`` gives one ``delta_theta`` series per ``bits_per_key``.
    """
    series: dict[str, list[tuple[object, str]]] = {}
    for variant in variants:
        if parameter not in variant.swept:
            continue
        others = ",".join(f"{key}={value}" for key, value in variant.swept.items() if key != parameter)
        key = f"{variant.family}[{others}]" if others else variant.family
        series.setdefault(key, []).append((variant.swept[parameter], variant.name))
    return series


def plot_sweep(
    results: Results,
    parameter: str,
    problems: list[str] | None = None,
    *,
    metric: DistributionMetric = "gap",
    columns: int = 2,
) -> Figure:
    """Median over seeds (line and markers) with the interquartile range (vertical bars).

    The x-axis is categorical, in the order the values are listed in the experiment, and is
    labelled with the values exactly as written (e.g. ``0.01pi``).
    """
    series = sweep_series(results.spec.variants, parameter)
    if not series:
        msg = f"no algorithm in this experiment sweeps {parameter!r} (swept: {results.spec.swept_parameters})"
        raise ValueError(msg)
    problems = problems or results.problems
    runs = run_metrics(results)
    column = "gap_percent" if metric == "gap" else "anytime_gap"
    colors = color_map(list(series))
    values = list(dict.fromkeys(str(value) for members in series.values() for value, _ in members))
    figure, axes = make_panels(len(problems), columns)

    for ax, problem in zip(axes, problems, strict=True):
        subset = runs[runs["problem"] == problem]
        offsets = np.linspace(-0.08, 0.08, len(series)) if len(series) > 1 else [0.0]
        for (key, members), offset in zip(series.items(), offsets, strict=True):
            x, low, mid, high = [], [], [], []
            for value, name in members:
                sample = subset.loc[subset["algorithm"] == name, column]
                if sample.empty:
                    continue
                x.append(values.index(str(value)) + offset)
                low.append(sample.quantile(0.25))
                mid.append(sample.median())
                high.append(sample.quantile(0.75))
            if not x:
                continue
            ax.vlines(x, low, high, color=colors[key], linewidth=LINE_WIDTH, alpha=0.45)
            ax.plot(x, mid, color=colors[key], linewidth=LINE_WIDTH, marker="o", markersize=8, markeredgecolor="white", markeredgewidth=1.5)
        style_axes(ax)
        ax.set_xticks(range(len(values)), values)
        ax.set_xlim(-0.5, len(values) - 0.5)
        ax.set_title(problem, fontsize=10, loc="left")
        ax.set_xlabel(parameter)
        ax.set_ylabel("final gap (%)" if metric == "gap" else "anytime gap (%)")

    series_legend(figure, colors)
    return figure
