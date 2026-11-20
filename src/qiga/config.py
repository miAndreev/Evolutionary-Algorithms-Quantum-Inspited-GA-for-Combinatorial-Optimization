"""Validated configuration models shared by all algorithms.

Algorithm-specific configs (classical GA, QIGA) extend :class:`GAConfig`.
"""

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
    elitism: NonNegativeInt = 1
    seed: NonNegativeInt | None = None

    @model_validator(mode="after")
    def _check_elitism(self) -> Self:
        if self.elitism >= self.population_size:
            msg = f"elitism ({self.elitism}) must be smaller than population_size ({self.population_size})"
            raise ValueError(msg)
        if self.max_evaluations is not None and self.max_evaluations < self.population_size:
            msg = f"max_evaluations ({self.max_evaluations}) must allow at least one population ({self.population_size})"
            raise ValueError(msg)
        return self
