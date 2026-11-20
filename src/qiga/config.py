"""Validated configuration models shared by all algorithms.

Algorithm-specific configs (classical GA, QIGA) extend :class:`GAConfig`.
"""

import math
from typing import Literal
from typing import Self

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import NonNegativeInt
from pydantic import PositiveInt
from pydantic import model_validator


class GAConfig(BaseModel):
    """Parameters common to every evolutionary algorithm in the framework.

    The run stops at whichever comes first: ``generations``, ``max_evaluations``
    (if set) or reaching ``target_fitness`` (if set).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    population_size: int = Field(default=50, ge=2)
    generations: PositiveInt = 100
    max_evaluations: PositiveInt | None = None
    target_fitness: float | None = None
    seed: NonNegativeInt | None = None
    repair: bool = False  # apply problem.repair() before evaluation (Lamarckian: the repaired genes are kept)

    @model_validator(mode="after")
    def _check_budget(self) -> Self:
        if self.max_evaluations is not None and self.max_evaluations < self.population_size:
            msg = f"max_evaluations ({self.max_evaluations}) must allow at least one population ({self.population_size})"
            raise ValueError(msg)
        return self


type SelectionName = Literal["tournament", "roulette"]
type CrossoverName = Literal["one_point", "two_point", "uniform", "ordered", "pmx"]
type MutationName = Literal["flip_bit", "shuffle", "inversion"]


class ClassicalGAConfig(GAConfig):
    """Generational GA with elitism.

    ``crossover``/``mutation`` left as ``None`` pick a sensible default for the problem's
    encoding. ``mutation_prob`` is the chance an offspring is mutated at all;
    ``gene_mutation_prob`` is the per-gene rate for ``flip_bit``/``shuffle`` (``None`` means 1/n).
    """

    elitism: NonNegativeInt = 1
    selection: SelectionName = "tournament"
    tournament_size: int = Field(default=3, ge=2)
    crossover: CrossoverName | None = None
    crossover_prob: float = Field(default=0.9, ge=0.0, le=1.0)
    mutation: MutationName | None = None
    mutation_prob: float = Field(default=1.0, ge=0.0, le=1.0)
    gene_mutation_prob: float | None = Field(default=None, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_elitism(self) -> Self:
        if self.elitism >= self.population_size:
            msg = f"elitism ({self.elitism}) must be smaller than population_size ({self.population_size})"
            raise ValueError(msg)
        return self


type RotationSchedule = Literal["constant", "linear_decay"]


class QIGAConfig(GAConfig):
    """Quantum-inspired GA (Han & Kim). Defaults follow their 0/1 knapsack setup.

    Each individual j keeps an attractor b_j (its best observed solution). Every generation
    its Q-chromosome is observed, and the Q-bits where the observation differs from b_j are
    rotated by ``delta_theta`` towards b_j (if the observation is worse) or by
    ``delta_theta_better`` towards the observation (if it is at least as good).

    * ``rotation_schedule="linear_decay"`` shrinks both angles linearly over the run,
      down to ``delta_theta_min / delta_theta`` of their initial size.
    * Local migration: within groups of ``local_group_size`` individuals, every attractor
      is replaced by the group's best, every generation (``1`` disables it).
    * Global migration: every ``global_migration_period`` generations all attractors are
      replaced by the global best (``None`` disables it).
    * ``epsilon > 0`` applies the H_epsilon gate; ``not_gate_prob > 0`` applies the quantum
      NOT gate to each Q-bit with that probability (a mutation operator).
    """

    population_size: int = Field(default=10, ge=2)
    delta_theta: float = Field(default=0.01 * math.pi, gt=0.0, le=math.pi / 2)
    delta_theta_better: float = Field(default=0.0, ge=0.0, le=math.pi / 2)
    rotation_schedule: RotationSchedule = "constant"
    delta_theta_min: float = Field(default=0.001 * math.pi, gt=0.0)
    local_group_size: PositiveInt = 2
    global_migration_period: PositiveInt | None = 100
    epsilon: float = Field(default=0.0, ge=0.0, lt=0.5)
    not_gate_prob: float = Field(default=0.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_qiga(self) -> Self:
        if self.delta_theta_min > self.delta_theta:
            msg = f"delta_theta_min ({self.delta_theta_min}) must not exceed delta_theta ({self.delta_theta})"
            raise ValueError(msg)
        if self.local_group_size > self.population_size:
            msg = f"local_group_size ({self.local_group_size}) must not exceed population_size ({self.population_size})"
            raise ValueError(msg)
        return self
