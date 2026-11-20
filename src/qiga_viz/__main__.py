"""Command line.

* ``python -m qiga_viz report <results>`` - every figure and statistics table for a results folder
* ``python -m qiga_viz demo tsp -n 30``   - trace one run: Q-bit heatmap, final tour, tour animation
"""

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

from qiga.algorithms.trace import run_traced
from qiga.experiments.analysis import load_results
from qiga.experiments.analysis import summary_table
from qiga.experiments.registry import ALGORITHMS
from qiga.experiments.stats import friedman_ranks
from qiga.experiments.stats import pairwise_tests
from qiga.problems.instances import KnapsackKind
from qiga.problems.instances import generate_euclidean_tsp
from qiga.problems.instances import generate_knapsack
from qiga.representations.decoders import RandomKeyDecoder
from qiga_viz.convergence import plot_convergence
from qiga_viz.distribution import plot_final_distribution
from qiga_viz.ecdf import plot_target_ecdf
from qiga_viz.style import SERIES_COLORS
from qiga_viz.sweep import plot_sweep
from qiga_viz.trace import animate_tours
from qiga_viz.trace import plot_qbit_heatmap
from qiga_viz.trace import plot_tour

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def _save(figure: Figure, folder: Path, name: str) -> None:
    for suffix in ("png", "pdf"):
        figure.savefig(folder / f"{name}.{suffix}", dpi=200)


def write_report(results_path: Path, output: Path | None = None) -> Path:
    """Write figures, the summary table and the statistical tests next to the results."""
    results = load_results(results_path)
    folder = output or results.folder / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    few_series = len(results.algorithms) <= len(SERIES_COLORS)
    if few_series:  # line plots need one distinct color per algorithm
        for metric in ("gap", "fitness"):
            _save(plot_convergence(results, metric=metric), folder, f"convergence_{metric}")
        _save(plot_target_ecdf(results), folder, "ecdf_targets")
    for metric in ("gap", "anytime"):
        _save(plot_final_distribution(results, metric=metric), folder, f"distribution_{metric}")
    for parameter in results.spec.swept_parameters:
        _save(plot_sweep(results, parameter), folder, f"sweep_{parameter}")

    summary_table(results).to_csv(folder / "summary.csv", index=False)
    for metric in ("gap", "anytime"):
        pairwise_tests(results, metric).to_csv(folder / f"stats_pairwise_{metric}.csv", index=False)
        ranks, p_value = friedman_ranks(results, metric)
        ranks.to_frame().assign(friedman_p=p_value).to_csv(folder / f"stats_ranks_{metric}.csv")
    return folder


def write_demo(problem_type: str, *, n: int, algorithm: str, evaluations: int, seed: int, output: Path) -> Path:
    """Run one algorithm with tracing and write its Q-bit heatmap (QIGA) and tour figures (TSP)."""
    problem = generate_euclidean_tsp(n, seed=seed) if problem_type == "tsp" else generate_knapsack(n, KnapsackKind.STRONGLY_CORRELATED, seed=seed)
    algorithm_cls, config_cls = ALGORITHMS[algorithm]
    config = config_cls(generations=10**9, max_evaluations=evaluations, seed=seed)
    result, trace = run_traced(algorithm_cls(problem, config))

    folder = output / f"{problem.name}-{algorithm}-s{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    if trace.frames[-1].p_one is not None:
        group = None
        if problem_type == "tsp":
            group = getattr(config, "bits_per_key", None) or RandomKeyDecoder.default_bits_per_key(n)
        _save(plot_qbit_heatmap(trace, group_size=group, title=f"{algorithm} on {problem.name}: P(observe 1) per Q-bit"), folder, "qbits")
    if problem_type == "tsp":
        _save(plot_tour(problem, result.best_solution, title=f"{algorithm}: best tour, length {result.best_fitness:.1f}"), folder, "tour")
        animate_tours(problem, trace, label=algorithm).save(folder / "tour.gif", writer="pillow", dpi=100)
    return folder


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="qiga_viz", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    report = commands.add_parser("report", help="write all figures, the summary table and statistical tests")
    report.add_argument("results", type=Path, help="results folder, or an experiment folder for its latest run")
    report.add_argument("-o", "--output", type=Path, default=None, help="figure folder (default: <results>/figures)")

    demo = commands.add_parser("demo", help="trace a single run and draw it")
    demo.add_argument("problem", choices=["tsp", "knapsack"])
    demo.add_argument("-n", type=int, default=30, help="cities or items (default: 30)")
    demo.add_argument("-a", "--algorithm", choices=sorted(ALGORITHMS), default="qiga")
    demo.add_argument("-e", "--evaluations", type=int, default=5000, help="evaluation budget (default: 5000)")
    demo.add_argument("-s", "--seed", type=int, default=0)
    demo.add_argument("-o", "--output", type=Path, default=Path("results/demo"))

    args = parser.parse_args(argv)
    if args.command == "report":
        print(write_report(args.results, args.output))
    else:
        print(write_demo(args.problem, n=args.n, algorithm=args.algorithm, evaluations=args.evaluations, seed=args.seed, output=args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
