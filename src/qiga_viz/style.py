"""Shared figure styling: a CVD-validated categorical palette and recessive axes."""

from typing import TYPE_CHECKING

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


def color_map(names: list[str]) -> dict[str, str]:
    """Color follows the entity: the n-th algorithm of an experiment always gets slot n."""
    if len(names) > len(SERIES_COLORS):
        msg = f"at most {len(SERIES_COLORS)} series can be colored distinctly, got {len(names)}"
        raise ValueError(msg)
    return dict(zip(names, SERIES_COLORS, strict=False))


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
