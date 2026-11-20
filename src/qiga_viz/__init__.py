"""Figures for qiga results folders and traced runs.

Experiment figures only read the files the runner writes; ``python -m qiga_viz demo`` runs a
single traced algorithm for the Q-bit heatmap and the tour animation.
"""

from qiga_viz.convergence import plot_convergence
from qiga_viz.distribution import plot_final_distribution
from qiga_viz.ecdf import plot_target_ecdf
from qiga_viz.sweep import plot_sweep
from qiga_viz.trace import animate_tours
from qiga_viz.trace import plot_qbit_heatmap
from qiga_viz.trace import plot_tour

__all__ = ["animate_tours", "plot_convergence", "plot_final_distribution", "plot_qbit_heatmap", "plot_sweep", "plot_target_ecdf", "plot_tour"]
