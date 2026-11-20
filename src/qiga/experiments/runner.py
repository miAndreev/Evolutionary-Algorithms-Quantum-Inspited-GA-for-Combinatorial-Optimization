"""Runs an experiment (problems x algorithms x seeds) and writes the results folder."""

from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING
from typing import NamedTuple

import pandas as pd

from qiga.experiments.registry import ALGORITHMS
from qiga.experiments.schema import CONVERGENCE_COLUMNS
from qiga.experiments.schema import HISTORY_COLUMNS
from qiga.experiments.schema import RUN_COLUMNS
from qiga.experiments.schema import ExperimentSpec
from qiga.experiments.schema import ProblemSpec
from qiga.experiments.schema import Variant

if TYPE_CHECKING:
    from qiga.algorithms.result import RunResult

type ProgressCallback = Callable[[int, int], None]


class _Task(NamedTuple):
    problem: ProblemSpec
    algorithm: Variant
    seed: int
    max_evaluations: int | None


def _execute(task: _Task) -> RunResult:
    problem = task.problem.build()
    algorithm_cls, _ = ALGORITHMS[task.algorithm.algorithm]
    config = task.algorithm.config_for(seed=task.seed, max_evaluations=task.max_evaluations)
    return algorithm_cls(problem, config).run()


def run_experiment(
    spec: ExperimentSpec,
    output_dir: Path | str,
    *,
    workers: int = 1,
    progress: ProgressCallback | None = None,
) -> Path:
    """Run every task and write ``<output_dir>/<spec.name>/<UTC timestamp>/``; returns that folder.

    Runs are independent and seeded, so the results do not depend on ``workers``.
    """
    tasks = [
        _Task(problem, algorithm, seed, spec.max_evaluations) for problem in spec.problems for algorithm in spec.variants for seed in spec.seed_list
    ]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(_with_progress(pool.map(_execute, tasks), len(tasks), progress))
    else:
        results = list(_with_progress(map(_execute, tasks), len(tasks), progress))

    references = {}
    for problem_spec in spec.problems:
        problem = problem_spec.build()
        references[problem.name] = (problem.sense.value, problem.known_optimum())

    run_rows, convergence_rows, history_rows = [], [], []
    for task, result in zip(tasks, results, strict=True):
        key = {"problem": result.problem, "algorithm": task.algorithm.name, "seed": task.seed}
        sense, reference = references[result.problem]
        run_rows.append(
            {
                **key,
                "sense": sense,
                "reference": reference,
                "best_fitness": result.best_fitness,
                "evaluations": result.evaluations,
                "generations": result.generations,
                "stop_reason": result.stop_reason.value,
                "evaluations_to_target": result.evaluations_to_target,
                "elapsed_seconds": result.elapsed_seconds,
            },
        )
        convergence_rows.extend({**key, **improvement.model_dump()} for improvement in result.improvements)
        history_rows.extend({**key, **record.model_dump()} for record in result.history)

    folder = Path(output_dir) / spec.name / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    folder.mkdir(parents=True)
    spec.to_yaml(folder / "experiment.yaml")
    _frame(run_rows, RUN_COLUMNS).to_parquet(folder / "runs.parquet", index=False)
    _frame(convergence_rows, CONVERGENCE_COLUMNS).to_parquet(folder / "convergence.parquet", index=False)
    _frame(history_rows, HISTORY_COLUMNS).to_parquet(folder / "history.parquet", index=False)
    return folder


def _with_progress[T](results: Iterable[T], total: int, progress: ProgressCallback | None) -> Iterator[T]:
    for done, result in enumerate(results, start=1):
        if progress is not None:
            progress(done, total)
        yield result


def _frame(rows: list[dict], columns: tuple[str, ...]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=list(columns))
    return frame.astype({"evaluations_to_target": "Int64"}) if "evaluations_to_target" in frame else frame
