"""Command line.

* ``python -m qiga run <experiment.yaml>``  - run an experiment, print the results folder
* ``python -m qiga summary <results>``      - summary table per problem and algorithm
* ``python -m qiga stats <results>``        - significance tests between the algorithms
"""

import argparse
from pathlib import Path
import sys

import pandas as pd

from qiga.experiments.analysis import load_results
from qiga.experiments.analysis import summary_table
from qiga.experiments.runner import run_experiment
from qiga.experiments.schema import ExperimentSpec
from qiga.experiments.stats import friedman_ranks
from qiga.experiments.stats import pairwise_tests


def _progress(done: int, total: int) -> None:
    print(f"\r{done}/{total} runs", end="" if done < total else "\n", file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="qiga", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="run an experiment and write a results folder")
    run.add_argument("spec", type=Path, help="experiment YAML file")
    run.add_argument("-o", "--output", type=Path, default=Path("results"), help="parent folder for results (default: results)")
    run.add_argument("-w", "--workers", type=int, default=1, help="parallel processes (default: 1)")

    summary = commands.add_parser("summary", help="print the summary table of a results folder")
    summary.add_argument("results", type=Path, help="results folder, or an experiment folder for its latest run")

    stats = commands.add_parser("stats", help="pairwise Mann-Whitney U tests (Holm-corrected) and Friedman ranks")
    stats.add_argument("results", type=Path, help="results folder, or an experiment folder for its latest run")
    stats.add_argument("-m", "--metric", choices=["gap", "anytime", "target"], default="gap", help="final gap, anytime gap or evaluations to target")
    stats.add_argument("-t", "--target", type=float, default=1.0, help="target gap in percent for --metric target (default: 1)")

    args = parser.parse_args(argv)
    if args.command == "run":
        folder = run_experiment(ExperimentSpec.from_yaml(args.spec), args.output, workers=args.workers, progress=_progress)
        print(folder)
        return 0
    results = load_results(args.results)
    with pd.option_context("display.width", 250, "display.max_columns", None, "display.max_colwidth", 60, "display.float_format", "{:.4g}".format):
        if args.command == "summary":
            print(summary_table(results).to_string(index=False))
        else:
            print(pairwise_tests(results, args.metric, target=args.target).to_string(index=False))
            ranks, p_value = friedman_ranks(results, args.metric, target=args.target)
            print(f"\nmean rank over {len(results.problems)} problems (1 = best), Friedman p = {'n/a' if p_value is None else f'{p_value:.4g}'}")
            print(ranks.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
