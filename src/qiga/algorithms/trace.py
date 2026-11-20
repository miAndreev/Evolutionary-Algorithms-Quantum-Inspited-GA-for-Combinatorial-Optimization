"""Record how a single run evolves (for figures and animations, not for experiments)."""

from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING

from qiga.algorithms.qiga import QIGA

if TYPE_CHECKING:
    import numpy as np

    from qiga.algorithms.base import EvolutionaryAlgorithm
    from qiga.algorithms.result import RunResult


@dataclass(frozen=True)
class TraceFrame:
    generation: int
    evaluations: int
    best_fitness: float
    best_solution: list[int]
    p_one: np.ndarray | None = None  # QIGA only: population-mean P(observe 1) of every Q-bit


@dataclass
class Trace:
    frames: list[TraceFrame] = field(default_factory=list)

    def improvements(self) -> list[TraceFrame]:
        """Frames where the best-so-far fitness changed (plus the first one)."""
        return [frame for i, frame in enumerate(self.frames) if i == 0 or frame.best_fitness != self.frames[i - 1].best_fitness]


def run_traced(algorithm: EvolutionaryAlgorithm, *, every: int = 1) -> tuple[RunResult, Trace]:
    """Run ``algorithm`` and snapshot it every ``every`` generations (and at the end).

    The run itself is identical to ``algorithm.run()``: the observer only reads state.
    """
    if every < 1:
        msg = "every must be a positive integer"
        raise ValueError(msg)
    trace = Trace()

    def observe(current: EvolutionaryAlgorithm) -> None:
        if current.generation % every:
            return
        trace.frames.append(_frame(current))

    result = algorithm.run(observe)
    if not trace.frames or trace.frames[-1].evaluations != algorithm.evaluator.evaluations:
        trace.frames.append(_frame(algorithm))
    return result, trace


def _frame(algorithm: EvolutionaryAlgorithm) -> TraceFrame:
    return TraceFrame(
        generation=algorithm.generation,
        evaluations=algorithm.evaluator.evaluations,
        best_fitness=algorithm.evaluator.best_fitness,
        best_solution=list(algorithm.evaluator.best_solution),
        p_one=algorithm.mean_p_one() if isinstance(algorithm, QIGA) else None,
    )
