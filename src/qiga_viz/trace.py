"""Figures for a single traced run: the Q-bit probability heatmap and TSP tours."""

from typing import TYPE_CHECKING

from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure
import numpy as np

from qiga_viz.style import GRID
from qiga_viz.style import PROBABILITY_CMAP
from qiga_viz.style import SERIES_COLORS
from qiga_viz.style import SURFACE
from qiga_viz.style import TEXT_PRIMARY
from qiga_viz.style import TEXT_SECONDARY
from qiga_viz.style import style_axes

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from qiga.algorithms.trace import Trace
    from qiga.problems.base import Solution
    from qiga.problems.tsp import TSP


def plot_qbit_heatmap(trace: Trace, *, group_size: int | None = None, title: str | None = None) -> Figure:
    """Population-mean P(observe 1) of every Q-bit (rows) over the run (columns).

    Gray is an undecided Q-bit (P = 1/2), blue has collapsed towards 1 and orange towards 0.
    ``group_size`` draws a separator every that many Q-bits (e.g. ``bits_per_key`` for TSP).
    """
    frames = [frame for frame in trace.frames if frame.p_one is not None]
    if not frames:
        msg = "the trace has no Q-bit probabilities (was it recorded from a QIGA run?)"
        raise ValueError(msg)
    matrix = np.column_stack([frame.p_one for frame in frames])
    evaluations = [frame.evaluations for frame in frames]

    n_bits = matrix.shape[0]
    figure = Figure(figsize=(8.0, min(2.0 + n_bits * 0.025, 9.0)), facecolor=SURFACE, layout="constrained")
    ax = figure.subplots()
    image = ax.imshow(
        matrix,
        aspect="auto",
        cmap=PROBABILITY_CMAP,
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        extent=(evaluations[0], evaluations[-1], n_bits - 0.5, -0.5),
    )
    if group_size:
        for boundary in range(group_size, n_bits, group_size):
            ax.axhline(boundary - 0.5, color=SURFACE, linewidth=0.6)
    style_axes(ax)
    ax.grid(visible=False)
    ax.set_xlabel("evaluations")
    ax.set_ylabel("Q-bit")
    ax.set_title(title or "Q-bit probabilities P(observe 1), population mean", fontsize=10, loc="left")
    colorbar = figure.colorbar(image, ax=ax, ticks=[0, 0.5, 1], shrink=0.8)
    colorbar.set_label("P(1)", color=TEXT_SECONDARY)
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(colors=TEXT_SECONDARY, length=0)
    return figure


def draw_tour(ax: Axes, problem: TSP, tour: Solution, *, color: str = SERIES_COLORS[0]) -> None:
    """Draw a closed tour on ``ax`` (cities as dots, the start city ringed)."""
    points = _coordinates(problem)
    closed = np.append(np.asarray(tour), tour[0])
    ax.plot(points[closed, 0], points[closed, 1], color=color, linewidth=1.6, zorder=2)
    ax.scatter(points[:, 0], points[:, 1], s=22, color=TEXT_PRIMARY, edgecolors=SURFACE, linewidths=1.2, zorder=3)
    ax.scatter(*points[tour[0]], s=90, facecolors="none", edgecolors=color, linewidths=1.6, zorder=3)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.set_facecolor(SURFACE)


def plot_tour(problem: TSP, tour: Solution, *, title: str | None = None) -> Figure:
    figure = Figure(figsize=(5.0, 5.0), facecolor=SURFACE, layout="constrained")
    ax = figure.subplots()
    draw_tour(ax, problem, tour)
    ax.set_title(title or f"length {problem.evaluate(tour):.1f}", fontsize=10, loc="left", color=TEXT_PRIMARY)
    return figure


def animate_tours(problem: TSP, trace: Trace, *, interval_ms: int = 250, label: str = "") -> FuncAnimation:
    """One frame per improvement of the best-so-far tour; save with ``.save(path, writer="pillow")``."""
    frames = trace.improvements()
    if not frames:
        msg = "the trace is empty"
        raise ValueError(msg)
    figure = Figure(figsize=(5.0, 5.4), facecolor=SURFACE, layout="constrained")
    ax = figure.subplots()

    def update(index: int) -> list:
        ax.clear()
        frame = frames[index]
        draw_tour(ax, problem, frame.best_solution)
        prefix = f"{label}  " if label else ""
        ax.set_title(f"{prefix}evaluations {frame.evaluations:,}   length {frame.best_fitness:.1f}", fontsize=10, loc="left", color=TEXT_PRIMARY)
        return []

    update(0)
    return FuncAnimation(figure, update, frames=len(frames), interval=interval_ms, repeat=False)


def _coordinates(problem: TSP) -> np.ndarray:
    if problem.coordinates is None:
        msg = "drawing a tour needs city coordinates (use TSP.from_coordinates or generate_euclidean_tsp)"
        raise ValueError(msg)
    return np.asarray(problem.coordinates)
