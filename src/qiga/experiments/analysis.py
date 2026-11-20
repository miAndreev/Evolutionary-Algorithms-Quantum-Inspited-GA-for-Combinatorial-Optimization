"""Loading a results folder and turning raw runs into comparable numbers."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from qiga.experiments.schema import RESULT_FILES
from qiga.experiments.schema import ExperimentSpec


@dataclass(frozen=True)
class Results:
    folder: Path
    spec: ExperimentSpec
    runs: pd.DataFrame
    convergence: pd.DataFrame
    history: pd.DataFrame

    @property
    def algorithms(self) -> list[str]:
        """Algorithm names in the order the experiment lists them (keeps colors stable)."""
        return [algorithm.name for algorithm in self.spec.algorithms]

    @property
    def problems(self) -> list[str]:
        return list(dict.fromkeys(self.runs["problem"]))


def load_results(path: Path | str) -> Results:
    """Load a results folder. An experiment folder loads its most recent run."""
    folder = Path(path)
    if not (folder / "runs.parquet").exists():
        candidates = sorted(child for child in folder.iterdir() if child.is_dir() and (child / "runs.parquet").exists()) if folder.is_dir() else []
        if not candidates:
            msg = f"no results found in {folder}"
            raise FileNotFoundError(msg)
        folder = candidates[-1]
    missing = [name for name in RESULT_FILES if not (folder / name).exists()]
    if missing:
        msg = f"results folder {folder} is missing {', '.join(missing)}"
        raise FileNotFoundError(msg)
    return Results(
        folder=folder,
        spec=ExperimentSpec.from_yaml(folder / "experiment.yaml"),
        runs=pd.read_parquet(folder / "runs.parquet"),
        convergence=pd.read_parquet(folder / "convergence.parquet"),
        history=pd.read_parquet(folder / "history.parquet"),
    )


def reference_values(runs: pd.DataFrame) -> pd.Series:
    """Per problem: the known optimum, or else the best fitness any run found."""

    def pick(group: pd.DataFrame) -> float:
        if group["reference"].notna().any():
            return float(group["reference"].dropna().iloc[0])
        maximize = group["sense"].iloc[0] == "maximize"
        return float(group["best_fitness"].max() if maximize else group["best_fitness"].min())

    return runs.groupby("problem", sort=False)[["reference", "sense", "best_fitness"]].apply(pick)


def gap_percent(fitness: pd.Series | np.ndarray, reference: float) -> np.ndarray:
    """Relative distance to the reference in percent; 0 means the reference was reached."""
    if reference == 0:
        return np.abs(np.asarray(fitness, dtype=np.float64)) * 100.0
    return np.abs(np.asarray(fitness, dtype=np.float64) - reference) / abs(reference) * 100.0


def best_so_far_on_grid(results: Results, problem: str, grid: np.ndarray) -> pd.DataFrame:
    """Each run's best-so-far fitness sampled at the evaluation counts in ``grid``.

    Returns columns ``algorithm, seed, evaluations, fitness``; ``fitness`` is NaN before a
    run's first evaluation and holds its final value after the run stopped.
    """
    rows = []
    subset = results.convergence[results.convergence["problem"] == problem]
    for (algorithm, seed), run in subset.groupby(["algorithm", "seed"], sort=False):
        evaluations = run["evaluations"].to_numpy()
        fitness = run["fitness"].to_numpy(dtype=np.float64)
        index = np.searchsorted(evaluations, grid, side="right") - 1
        values = np.where(index >= 0, fitness[np.clip(index, 0, None)], np.nan)
        rows.append(pd.DataFrame({"algorithm": algorithm, "seed": seed, "evaluations": grid, "fitness": values}))
    return pd.concat(rows, ignore_index=True)


def summary_table(results: Results) -> pd.DataFrame:
    """One row per problem x algorithm with final-quality and cost statistics."""
    runs = results.runs.copy()
    references = reference_values(runs)
    runs["gap_percent"] = [
        gap_percent(np.array([fitness]), references[problem])[0] for problem, fitness in zip(runs["problem"], runs["best_fitness"], strict=True)
    ]
    runs["hit_reference"] = runs["gap_percent"] <= 1e-9
    table = runs.groupby(["problem", "algorithm"], sort=False).agg(
        runs=("seed", "count"),
        best_mean=("best_fitness", "mean"),
        best_std=("best_fitness", "std"),
        best_median=("best_fitness", "median"),
        gap_mean_percent=("gap_percent", "mean"),
        hit_rate=("hit_reference", "mean"),
        evaluations_mean=("evaluations", "mean"),
        seconds_mean=("elapsed_seconds", "mean"),
    )
    table.insert(0, "reference", [references[problem] for problem, _ in table.index])
    return table.reset_index()
