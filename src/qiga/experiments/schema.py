"""Experiment definitions (loaded from YAML) and the on-disk results format.

A results folder contains:

* ``experiment.yaml``    - the validated :class:`ExperimentSpec` that produced it
* ``runs.parquet``       - one row per run, columns :data:`RUN_COLUMNS`
* ``convergence.parquet``- best-so-far improvements, columns :data:`CONVERGENCE_COLUMNS`
* ``history.parquet``    - per-generation statistics, columns :data:`HISTORY_COLUMNS`
"""

from pathlib import Path
from typing import TYPE_CHECKING
from typing import Annotated
from typing import Any
from typing import Literal
from typing import Self

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import PositiveFloat
from pydantic import PositiveInt
from pydantic import model_validator
import yaml

from qiga.experiments.registry import ALGORITHMS
from qiga.problems.instances import KnapsackKind
from qiga.problems.instances import generate_euclidean_tsp
from qiga.problems.instances import generate_knapsack

if TYPE_CHECKING:
    from qiga.config import GAConfig
    from qiga.problems.base import Problem

UNLIMITED_GENERATIONS = 10**9
RESULT_FILES = ("experiment.yaml", "runs.parquet", "convergence.parquet", "history.parquet")
RUN_COLUMNS = (
    "problem",
    "sense",
    "reference",
    "algorithm",
    "seed",
    "best_fitness",
    "evaluations",
    "generations",
    "stop_reason",
    "evaluations_to_target",
    "elapsed_seconds",
)
CONVERGENCE_COLUMNS = ("problem", "algorithm", "seed", "evaluations", "fitness")
HISTORY_COLUMNS = ("problem", "algorithm", "seed", "generation", "evaluations", "best_so_far", "best", "mean", "std", "worst")


class _Spec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class KnapsackSpec(_Spec):
    type: Literal["knapsack"] = "knapsack"
    n: PositiveInt
    correlation: KnapsackKind = KnapsackKind.UNCORRELATED
    max_weight: PositiveInt = 1000
    capacity_ratio: float = Field(default=0.5, gt=0, lt=1)
    seed: int = 0

    def build(self) -> Problem:
        return generate_knapsack(self.n, self.correlation, max_weight=self.max_weight, capacity_ratio=self.capacity_ratio, seed=self.seed)


class TSPSpec(_Spec):
    type: Literal["tsp"] = "tsp"
    n: int = Field(ge=2)
    size: PositiveFloat = 1000.0
    rounded: bool = False
    seed: int = 0

    def build(self) -> Problem:
        return generate_euclidean_tsp(self.n, size=self.size, rounded=self.rounded, seed=self.seed)


type ProblemSpec = Annotated[KnapsackSpec | TSPSpec, Field(discriminator="type")]


class AlgorithmSpec(_Spec):
    """One algorithm variant. ``params`` are validated against the algorithm's config model."""

    algorithm: str
    label: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.algorithm not in ALGORITHMS:
            msg = f"unknown algorithm {self.algorithm!r} (available: {', '.join(sorted(ALGORITHMS))})"
            raise ValueError(msg)
        if "seed" in self.params:
            msg = "set seeds on the experiment, not in algorithm params"
            raise ValueError(msg)
        self.config_for(seed=0)
        return self

    @property
    def name(self) -> str:
        return self.label or self.algorithm

    def config_for(self, *, seed: int, max_evaluations: int | None = None) -> GAConfig:
        overrides: dict[str, Any] = {"seed": seed}
        if max_evaluations is not None:
            overrides["max_evaluations"] = max_evaluations
            if "generations" not in self.params:
                overrides["generations"] = UNLIMITED_GENERATIONS
        _, config_cls = ALGORITHMS[self.algorithm]
        return config_cls.model_validate({**self.params, **overrides})


class ExperimentSpec(_Spec):
    """Every algorithm runs on every problem once per seed.

    ``max_evaluations`` (if set) is applied to every algorithm so they get the same budget;
    the generation limit is then lifted unless an algorithm sets ``generations`` itself.
    """

    name: str = Field(pattern=r"^[\w.-]+$")
    seeds: list[int] | PositiveInt = 10
    max_evaluations: PositiveInt | None = None
    problems: list[ProblemSpec] = Field(min_length=1)
    algorithms: list[AlgorithmSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_unique_names(self) -> Self:
        names = [algorithm.name for algorithm in self.algorithms]
        if len(set(names)) != len(names):
            msg = f"algorithm names must be unique, use 'label' to distinguish variants: {names}"
            raise ValueError(msg)
        return self

    @property
    def seed_list(self) -> list[int]:
        return list(range(self.seeds)) if isinstance(self.seeds, int) else list(self.seeds)

    @classmethod
    def from_yaml(cls, path: Path | str) -> Self:
        return cls.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))

    def to_yaml(self, path: Path | str) -> None:
        Path(path).write_text(yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False), encoding="utf-8")
