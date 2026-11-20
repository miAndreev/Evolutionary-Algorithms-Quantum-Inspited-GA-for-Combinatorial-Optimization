"""Command line: ``python -m qiga run <experiment.yaml>`` and ``python -m qiga summary <results>``."""

import argparse
from pathlib import Path
import sys

import pandas as pd

from qiga.experiments.analysis import load_results
from qiga.experiments.analysis import summary_table
from qiga.experiments.runner import run_experiment
from qiga.experiments.schema import ExperimentSpec


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

    args = parser.parse_args(argv)
    if args.command == "run":
        folder = run_experiment(ExperimentSpec.from_yaml(args.spec), args.output, workers=args.workers, progress=_progress)
        print(folder)
    else:
        with pd.option_context("display.width", 200, "display.max_columns", None, "display.float_format", "{:.4g}".format):
            print(summary_table(load_results(args.results)).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
