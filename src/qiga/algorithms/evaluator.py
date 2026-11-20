"""Fitness evaluation with budget enforcement and convergence tracking."""

import numpy as np

from qiga.problems.base import Problem
from qiga.problems.base import Sense
from qiga.problems.base import Solution


class BudgetExhaustedError(RuntimeError):
    """Raised when an evaluation is requested after the evaluation budget is spent."""


class Evaluator:
    """Counts evaluations and remembers the best solution ever evaluated.

    Every algorithm evaluates solutions only through an ``Evaluator``. This makes the
    evaluation count and the best-so-far trace identical in meaning for all algorithms,
    which is what a fair convergence-speed comparison needs.
    """

    def __init__(self, problem: Problem, *, max_evaluations: int | None = None, target_fitness: float | None = None) -> None:
        self.problem = problem
        self.max_evaluations = max_evaluations
        self.target_fitness = target_fitness
        self.evaluations = 0
        self.best_fitness: float | None = None
        self.best_solution: list[int] | None = None
        self.improvements: list[tuple[int, float]] = []  # (evaluation number, new best fitness)
        self.target_hit_at: int | None = None

    def __call__(self, solution: Solution) -> float:
        if self.budget_exhausted:
            msg = f"evaluation budget of {self.max_evaluations} is exhausted"
            raise BudgetExhaustedError(msg)
        fitness = self.problem.evaluate(solution)
        self.evaluations += 1
        if self.best_fitness is None or self.problem.is_better(fitness, self.best_fitness):
            self.best_fitness = fitness
            self.best_solution = np.asarray(solution, dtype=np.int64).tolist()
            self.improvements.append((self.evaluations, fitness))
            if self.target_hit_at is None and self._reaches_target(fitness):
                self.target_hit_at = self.evaluations
        return fitness

    @property
    def remaining(self) -> int | None:
        return None if self.max_evaluations is None else self.max_evaluations - self.evaluations

    @property
    def budget_exhausted(self) -> bool:
        return self.max_evaluations is not None and self.evaluations >= self.max_evaluations

    @property
    def target_reached(self) -> bool:
        return self.target_hit_at is not None

    def _reaches_target(self, fitness: float) -> bool:
        if self.target_fitness is None:
            return False
        if self.problem.sense is Sense.MAXIMIZE:
            return fitness >= self.target_fitness
        return fitness <= self.target_fitness
