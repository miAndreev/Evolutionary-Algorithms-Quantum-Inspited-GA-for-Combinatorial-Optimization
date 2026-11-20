"""Serializable results of a single algorithm run."""

from enum import StrEnum
from typing import Any

import pandas as pd
from pydantic import BaseModel
from pydantic import ConfigDict


class StopReason(StrEnum):
    GENERATIONS = "generations"
    EVALUATIONS = "evaluations"
    TARGET = "target"


class GenerationRecord(BaseModel):
    """Statistics recorded after each generation (generation 0 is the initial population).

    ``best``/``mean``/``std``/``worst`` describe the current population;
    ``best_so_far`` is the best fitness over all evaluations up to this point.
    """

    model_config = ConfigDict(frozen=True)

    generation: int
    evaluations: int
    best_so_far: float
    best: float
    mean: float
    std: float
    worst: float


class Improvement(BaseModel):
    """The best-so-far fitness changed to ``fitness`` at evaluation number ``evaluations``."""

    model_config = ConfigDict(frozen=True)

    evaluations: int
    fitness: float


class RunResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    algorithm: str
    problem: str | None
    config: dict[str, Any]
    best_solution: list[int]
    best_fitness: float
    evaluations: int
    generations: int
    stop_reason: StopReason
    evaluations_to_target: int | None
    elapsed_seconds: float
    history: list[GenerationRecord]
    improvements: list[Improvement]

    def history_frame(self) -> pd.DataFrame:
        """Per-generation statistics as a DataFrame."""
        return pd.DataFrame([record.model_dump() for record in self.history])

    def convergence_frame(self) -> pd.DataFrame:
        """Best-so-far fitness as a step function of the evaluation count."""
        return pd.DataFrame([improvement.model_dump() for improvement in self.improvements])
