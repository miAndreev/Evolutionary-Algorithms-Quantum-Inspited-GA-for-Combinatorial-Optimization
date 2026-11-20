"""Template for all evolutionary algorithms in the framework."""

from abc import ABC
from abc import abstractmethod
from time import perf_counter
from typing import TYPE_CHECKING
from typing import ClassVar

import numpy as np

from qiga.algorithms.evaluator import BudgetExhaustedError
from qiga.algorithms.evaluator import Evaluator
from qiga.algorithms.result import GenerationRecord
from qiga.algorithms.result import Improvement
from qiga.algorithms.result import RunResult
from qiga.algorithms.result import StopReason
from qiga.config import GAConfig
from qiga.problems.base import Problem
from qiga.problems.base import Sense

if TYPE_CHECKING:
    from collections.abc import Sequence


class EvolutionaryAlgorithm[C: GAConfig](ABC):
    """Runs the generational loop; subclasses define only how a population is created and evolved.

    Subclasses must evaluate solutions exclusively through ``self.evaluate`` and draw all
    randomness from ``self.rng`` so that runs are counted fairly and are reproducible.

    If the evaluation budget runs out in the middle of a generation, the run stops at once.
    The partially built generation is discarded, but every solution evaluated in it still
    counts towards the best-so-far result.
    """

    name: ClassVar[str]

    def __init__(self, problem: Problem, config: C) -> None:
        self.problem = problem
        self.config = config
        self._reset()

    @abstractmethod
    def _initialize(self) -> None:
        """Create and evaluate the initial population."""

    @abstractmethod
    def _step(self) -> None:
        """Advance the population by one generation."""

    @abstractmethod
    def _population_fitness(self) -> Sequence[float]:
        """Fitness values of the current population, used for per-generation statistics."""

    def run(self) -> RunResult:
        """Run from scratch; calling ``run`` again with the same seed reproduces the result."""
        self._reset()
        start = perf_counter()
        try:
            self._initialize()
            self._record()
            while (reason := self._stop_reason()) is None:
                self.generation += 1
                self._step()
                self._record()
        except BudgetExhaustedError:
            reason = StopReason.EVALUATIONS
            if self.history:  # budget ran out mid-generation; log the evaluations it consumed
                self._record()
        if not self.history:
            msg = "the evaluation budget is too small to evaluate the initial population"
            raise ValueError(msg)
        return self._result(reason, perf_counter() - start)

    def evaluate(self, solution: Sequence[int] | np.ndarray) -> float:
        return self.evaluator(solution)

    def prepare(self, solution: Sequence[int] | np.ndarray) -> list[int]:
        """The solution that will actually be evaluated: repaired if ``config.repair`` is set."""
        if self.config.repair:
            return self.problem.repair(solution)
        return np.asarray(solution, dtype=np.int64).tolist()

    def _reset(self) -> None:
        self.rng = np.random.default_rng(self.config.seed)
        self.evaluator = Evaluator(self.problem, max_evaluations=self.config.max_evaluations, target_fitness=self.config.target_fitness)
        self.generation = 0
        self.history: list[GenerationRecord] = []

    def _stop_reason(self) -> StopReason | None:
        if self.evaluator.target_reached:
            return StopReason.TARGET
        if self.evaluator.budget_exhausted:
            return StopReason.EVALUATIONS
        if self.generation >= self.config.generations:
            return StopReason.GENERATIONS
        return None

    def _record(self) -> None:
        if self.history and self.history[-1].evaluations == self.evaluator.evaluations:
            return  # nothing new happened since the last record
        fitness = np.asarray(self._population_fitness(), dtype=np.float64)
        maximize = self.problem.sense is Sense.MAXIMIZE
        record = GenerationRecord(
            generation=self.generation,
            evaluations=self.evaluator.evaluations,
            best_so_far=self.evaluator.best_fitness,
            best=fitness.max() if maximize else fitness.min(),
            mean=fitness.mean(),
            std=fitness.std(),
            worst=fitness.min() if maximize else fitness.max(),
        )
        self.history.append(record)

    def _result(self, reason: StopReason, elapsed: float) -> RunResult:
        return RunResult(
            algorithm=self.name,
            problem=self.problem.name,
            config=self.config.model_dump(mode="json"),
            best_solution=self.evaluator.best_solution,
            best_fitness=self.evaluator.best_fitness,
            evaluations=self.evaluator.evaluations,
            generations=self.generation,
            stop_reason=reason,
            evaluations_to_target=self.evaluator.target_hit_at,
            elapsed_seconds=elapsed,
            history=self.history,
            improvements=[Improvement(evaluations=e, fitness=f) for e, f in self.evaluator.improvements],
        )
