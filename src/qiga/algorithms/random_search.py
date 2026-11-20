"""Random search baseline: every generation is a fresh batch of random solutions."""

from typing import TYPE_CHECKING
from typing import ClassVar

from qiga.algorithms.base import EvolutionaryAlgorithm
from qiga.config import GAConfig

if TYPE_CHECKING:
    from collections.abc import Sequence


class RandomSearch(EvolutionaryAlgorithm[GAConfig]):
    """Lower baseline for comparisons: any real optimizer should beat it at equal budget."""

    name: ClassVar[str] = "random_search"

    def _initialize(self) -> None:
        self._sample()

    def _step(self) -> None:
        self._sample()

    def _population_fitness(self) -> Sequence[float]:
        return self._fitness

    def _sample(self) -> None:
        self._fitness = [self.evaluate(self.problem.random_solution(self.rng)) for _ in range(self.config.population_size)]
