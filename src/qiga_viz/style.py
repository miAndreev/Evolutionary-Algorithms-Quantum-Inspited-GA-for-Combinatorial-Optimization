"""Shared figure styling: a CVD-validated categorical palette and recessive axes."""

from typing import TYPE_CHECKING

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

if TYPE_CHECKING:
    from matplotlib.axes import Axes

# Categorical slots in fixed order (validated: adjacent pairs separable under colour-vision deficiency).
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"

LINE_WIDTH = 2.0
BAND_ALPHA = 0.16
NEUTRAL = "#8a8985"  # marks whose identity is given by an axis label rather than a color

# Diverging map for probabilities: P(1) = 0 orange, 1/2 neutral gray (undecided), 1 blue.
PROBABILITY_CMAP = LinearSegmentedColormap.from_list("p_one", [SERIES_COLORS[1], "#d9d8d4", SERIES_COLORS[0]])

_LABEL_MIN_SPACING = 0.06  # in axes-height fractions


def color_map(names: list[str]) -> dict[str, str]:
    """Color follows the entity: the n-th algorithm of an experiment always gets slot n."""
    if len(names) > len(SERIES_COLORS):
        msg = f"at most {len(SERIES_COLORS)} series can be colored distinctly, got {len(names)}"
        raise ValueError(msg)
    return dict(zip(names, SERIES_COLORS, strict=False))


def colors_or_neutral(names: list[str]) -> dict[str, str]:
    """:func:`color_map`, or one neutral color when there are too many series to color apart."""
    return color_map(names) if len(names) <= len(SERIES_COLORS) else dict.fromkeys(names, NEUTRAL)


def make_panels(count: int, columns: int = 2, *, width: float = 5.2, height: float = 3.6) -> tuple[Figure, list[Axes]]:
    """A figure with ``count`` visible panels laid out in rows of ``columns``."""
    columns = max(1, min(columns, count))
    rows = -(-count // columns)
    figure = Figure(figsize=(width * columns, height * rows + 0.6), facecolor=SURFACE, layout="constrained")
    axes = list(figure.subplots(rows, columns, squeeze=False).ravel())
    for ax in axes[count:]:
        ax.set_visible(False)
    return figure, axes[:count]


def label_line_ends(ax: Axes, ends: list[tuple[float, str]], colors: dict[str, str]) -> None:
    """Direct labels at the right edge for ``(y value, name)`` pairs, nudged apart so they never overlap."""
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


def series_legend(figure: Figure, colors: dict[str, str]) -> None:
    """One legend row above all panels."""
    handles = [Line2D([], [], color=color, linewidth=LINE_WIDTH, label=name) for name, color in colors.items()]
    figure.legend(handles=handles, loc="outside upper left", ncols=min(len(handles), 4), frameon=False, labelcolor=TEXT_PRIMARY, fontsize=9)


def style_axes(ax: Axes) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(visible=True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9, length=0)
    ax.xaxis.label.set_color(TEXT_SECONDARY)
    ax.yaxis.label.set_color(TEXT_SECONDARY)
    ax.title.set_color(TEXT_PRIMARY)
