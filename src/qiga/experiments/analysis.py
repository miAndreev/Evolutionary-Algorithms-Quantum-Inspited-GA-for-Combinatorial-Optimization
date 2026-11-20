"""Loading a results folder and turning raw runs into comparable numbers."""

from dataclasses import dataclass
from functools import cached_property
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
        return [variant.name for variant in self.spec.variants]

    @property
    def problems(self) -> list[str]:
        return list(dict.fromkeys(self.runs["problem"]))

    @cached_property
    def runs_by_key(self) -> pd.DataFrame:
        """``runs`` indexed by ``(problem, algorithm, seed)``."""
        return self.runs.set_index(["problem", "algorithm", "seed"])


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


def run_metrics(results: Results, targets: tuple[float, ...] = ()) -> pd.DataFrame:
    """One row per run with comparable quality and speed measures.

    * ``gap_percent`` - final distance to the reference (see :func:`reference_values`);
    * ``anytime_gap`` - mean gap over the whole budget (area under the best-so-far gap curve
      divided by the evaluations used): rewards finding good solutions *early*;
    * ``evals_to_<t>`` - evaluations until the gap first dropped to ``t`` percent or below
      (``NaN`` if the run never got there), for every ``t`` in ``targets``.
    """
    runs = results.runs.copy()
    references = reference_values(runs)
    runs["gap_percent"] = [gap_percent(np.array([f]), references[p])[0] for p, f in zip(runs["problem"], runs["best_fitness"], strict=True)]
    anytime: dict[tuple, float] = {}
    reached: dict[float, dict[tuple, float]] = {target: {} for target in targets}
    for key, run in results.convergence.groupby(["problem", "algorithm", "seed"], sort=False):
        evaluations = run["evaluations"].to_numpy(dtype=np.float64)
        gaps = gap_percent(run["fitness"], references[key[0]])
        budget = max(float(results.runs_by_key.loc[key, "evaluations"]), evaluations[-1])
        widths = np.diff(np.append(evaluations, budget + 1))  # each gap holds until the next improvement
        anytime[key] = float((gaps * widths).sum() / widths.sum())
        for target in targets:
            hits = np.flatnonzero(gaps <= target + 1e-9)
            reached[target][key] = evaluations[hits[0]] if hits.size else np.nan
    keys = list(zip(runs["problem"], runs["algorithm"], runs["seed"], strict=True))
    runs["anytime_gap"] = [anytime.get(key, np.nan) for key in keys]
    for target in targets:
        runs[target_column(target)] = [reached[target].get(key, np.nan) for key in keys]
    return runs


def target_column(target: float) -> str:
    return f"evals_to_{target:g}"


def summary_table(results: Results) -> pd.DataFrame:
    """One row per problem x algorithm with final-quality, speed and cost statistics."""
    runs = run_metrics(results)
    runs["hit_reference"] = runs["gap_percent"] <= 1e-9
    table = runs.groupby(["problem", "algorithm"], sort=False).agg(
        runs=("seed", "count"),
        best_mean=("best_fitness", "mean"),
        best_std=("best_fitness", "std"),
        best_median=("best_fitness", "median"),
        gap_mean_percent=("gap_percent", "mean"),
        anytime_gap_mean=("anytime_gap", "mean"),
        hit_rate=("hit_reference", "mean"),
        evaluations_mean=("evaluations", "mean"),
        seconds_mean=("elapsed_seconds", "mean"),
    )
    table.insert(0, "reference", reference_values(results.runs).reindex([problem for problem, _ in table.index]).to_numpy())
    return table.reset_index()
