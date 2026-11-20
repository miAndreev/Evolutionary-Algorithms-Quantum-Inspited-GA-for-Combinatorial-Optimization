"""Statistical comparison of algorithms over independent seeds.

All metrics are "lower is better" (gaps, anytime gaps, evaluations to a target), so the
tests are oriented that way: ``a12 > 0.5`` means the first algorithm tends to be better.

* Pairwise, per problem: two-sided Mann-Whitney U test with Holm's correction over all
  pairs of that problem, plus the Vargha-Delaney A12 effect size.
* Across problems: Friedman test on the per-problem medians and the mean rank of each
  algorithm (rank 1 = best).
"""

import itertools
from typing import TYPE_CHECKING
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

from qiga.experiments.analysis import run_metrics
from qiga.experiments.analysis import target_column

if TYPE_CHECKING:
    from collections.abc import Sequence

    from qiga.experiments.analysis import Results

type Metric = Literal["gap", "anytime", "target"]

ALPHA = 0.05
# Vargha & Delaney (2000) thresholds on |A12 - 0.5|.
_EFFECT_SIZES = ((0.06, "negligible"), (0.14, "small"), (0.21, "medium"))


def metric_values(results: Results, metric: Metric = "gap", *, target: float = 1.0) -> pd.DataFrame:
    """Columns ``problem, algorithm, seed, value`` for the chosen lower-is-better metric.

    ``target`` uses the evaluations needed to reach a gap of ``target`` percent; runs that
    never did count as needing infinitely many (they rank last, which the rank tests handle).
    """
    if metric == "target":
        runs = run_metrics(results, targets=(target,))
        values = runs[target_column(target)].fillna(np.inf)
    else:
        runs = run_metrics(results)
        values = runs["gap_percent" if metric == "gap" else "anytime_gap"]
    return runs[["problem", "algorithm", "seed"]].assign(value=values.to_numpy(dtype=np.float64))


def vargha_delaney_a12(a: Sequence[float], b: Sequence[float]) -> float:
    """Probability that a random run of ``a`` has a lower value than one of ``b`` (ties count half)."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        msg = "both samples need at least one value"
        raise ValueError(msg)
    less = (a[:, None] < b[None, :]).sum()
    equal = (a[:, None] == b[None, :]).sum()
    return float((less + 0.5 * equal) / (a.size * b.size))


def effect_magnitude(a12: float) -> str:
    distance = abs(a12 - 0.5)
    return next((label for limit, label in _EFFECT_SIZES if distance < limit), "large")


def holm(p_values: Sequence[float]) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (same order as the input)."""
    p = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (p.size - rank) * p[index]))
        adjusted[index] = running
    return adjusted


def pairwise_tests(results: Results, metric: Metric = "gap", *, target: float = 1.0, alpha: float = ALPHA) -> pd.DataFrame:
    """Every pair of algorithms on every problem.

    ``winner`` is the better algorithm when the Holm-adjusted p-value is below ``alpha``,
    otherwise ``"-"`` (no significant difference).
    """
    values = metric_values(results, metric, target=target)
    rows = []
    for problem in results.problems:
        subset = values[values["problem"] == problem]
        samples = {name: subset.loc[subset["algorithm"] == name, "value"].to_numpy() for name in results.algorithms}
        samples = {name: sample for name, sample in samples.items() if sample.size}
        problem_rows = []
        for first, second in itertools.combinations(samples, 2):
            a, b = samples[first], samples[second]
            p = stats.mannwhitneyu(a, b).pvalue
            p = 1.0 if np.isnan(p) else p  # e.g. every value identical
            a12 = vargha_delaney_a12(a, b)
            problem_rows.append(
                {
                    "problem": problem,
                    "a": first,
                    "b": second,
                    "median_a": float(np.median(a)),
                    "median_b": float(np.median(b)),
                    "p_value": float(p),
                    "a12": a12,
                    "effect": effect_magnitude(a12),
                },
            )
        for row, adjusted in zip(problem_rows, holm([row["p_value"] for row in problem_rows]), strict=True):
            row["p_holm"] = float(adjusted)
            row["winner"] = (row["a"] if row["a12"] > 0.5 else row["b"]) if adjusted < alpha else "-"
        rows.extend(problem_rows)
    columns = ["problem", "a", "b", "median_a", "median_b", "p_value", "p_holm", "a12", "effect", "winner"]
    return pd.DataFrame(rows, columns=columns)


def friedman_ranks(results: Results, metric: Metric = "gap", *, target: float = 1.0) -> tuple[pd.Series, float | None]:
    """Mean rank of each algorithm over problems (by median value) and the Friedman p-value.

    The p-value is ``None`` when the test is undefined (fewer than 2 problems or 3 algorithms).
    """
    values = metric_values(results, metric, target=target)
    medians = values.pivot_table(index="problem", columns="algorithm", values="value", aggfunc="median", sort=False)
    medians = medians.reindex(columns=[name for name in results.algorithms if name in medians.columns]).dropna()
    ranks = medians.rank(axis=1, method="average").mean().rename("mean_rank")
    minimum_algorithms = 3
    if len(medians) < 2 or medians.shape[1] < minimum_algorithms:
        return ranks, None
    if (medians.max(axis=1) == medians.min(axis=1)).all():
        return ranks, 1.0
    return ranks, float(stats.friedmanchisquare(*(medians[column] for column in medians.columns)).pvalue)
