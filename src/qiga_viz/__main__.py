"""Command line: ``python -m qiga_viz report <results>`` writes all figures next to the results."""

import argparse
from pathlib import Path

from qiga.experiments.analysis import load_results
from qiga.experiments.analysis import summary_table
from qiga_viz.convergence import plot_convergence


def write_report(results_path: Path, output: Path | None = None) -> Path:
    results = load_results(results_path)
    folder = output or results.folder / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    for metric in ("gap", "fitness"):
        figure = plot_convergence(results, metric=metric)
        for suffix in ("png", "pdf"):
            figure.savefig(folder / f"convergence_{metric}.{suffix}", dpi=200)
    summary_table(results).to_csv(folder / "summary.csv", index=False)
    return folder


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="qiga_viz", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    report = commands.add_parser("report", help="write convergence figures and a summary table")
    report.add_argument("results", type=Path, help="results folder, or an experiment folder for its latest run")
    report.add_argument("-o", "--output", type=Path, default=None, help="figure folder (default: <results>/figures)")
    args = parser.parse_args(argv)
    print(write_report(args.results, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
