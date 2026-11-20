"""Classical generational GA built on DEAP operators."""

from copy import deepcopy
import random
from typing import TYPE_CHECKING
from typing import ClassVar

from deap import base
from deap import tools

from qiga.algorithms.base import EvolutionaryAlgorithm
from qiga.config import ClassicalGAConfig
from qiga.operators.registry import make_crossover
from qiga.operators.registry import make_mutation
from qiga.operators.registry import make_selection
from qiga.problems.base import Problem
from qiga.problems.base import Sense

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Sequence


class FitnessMax(base.Fitness):
    weights = (1.0,)


class FitnessMin(base.Fitness):
    weights = (-1.0,)


class Individual(list[int]):
    """A list of genes with a DEAP fitness attached (what ``creator.create`` would build)."""

    def __init__(self, genes: Iterable[int], fitness: base.Fitness) -> None:
        super().__init__(genes)
        self.fitness = fitness


class ClassicalGA(EvolutionaryAlgorithm[ClassicalGAConfig]):
    """Generational GA: elitism + selection + crossover + mutation, operators chosen by encoding.

    Only offspring whose genes changed are re-evaluated, so a generation costs at most
    ``population_size - elitism`` evaluations.
    """

    name: ClassVar[str] = "classical_ga"

    def __init__(self, problem: Problem, config: ClassicalGAConfig) -> None:
        super().__init__(problem, config)
        gene_prob = config.gene_mutation_prob or 1.0 / problem.dimension
        self._fitness_cls = FitnessMax if problem.sense is Sense.MAXIMIZE else FitnessMin
        self.toolbox = base.Toolbox()
        self.toolbox.register("mate", make_crossover(config.crossover, problem.encoding))
        self.toolbox.register("mutate", make_mutation(config.mutation, problem.encoding, gene_prob))
        self.toolbox.register("select", make_selection(config.selection, problem.sense, config.tournament_size))
        self.population: list[Individual] = []

    def _reset(self) -> None:
        super()._reset()
        random.seed(self.config.seed)  # DEAP operators use the global random module

    def _initialize(self) -> None:
        population = [Individual(self.problem.random_solution(self.rng), self._fitness_cls()) for _ in range(self.config.population_size)]
        self._evaluate_invalid(population)
        self.population = population

    def _step(self) -> None:
        # Clone one by one: deepcopy of the whole list would keep a parent selected twice as a single shared object.
        elite = [deepcopy(ind) for ind in tools.selBest(self.population, self.config.elitism)]
        offspring = [deepcopy(ind) for ind in self.toolbox.select(self.population, self.config.population_size - self.config.elitism)]

        for first, second in zip(offspring[::2], offspring[1::2], strict=False):
            if random.random() < self.config.crossover_prob:  # noqa: S311
                self.toolbox.mate(first, second)
                del first.fitness.values, second.fitness.values
        for individual in offspring:
            if random.random() < self.config.mutation_prob:  # noqa: S311
                self.toolbox.mutate(individual)
                del individual.fitness.values

        self._evaluate_invalid(offspring)
        self.population = elite + offspring

    def _population_fitness(self) -> Sequence[float]:
        return [individual.fitness.values[0] for individual in self.population]

    def _evaluate_invalid(self, individuals: list[Individual]) -> None:
        for individual in individuals:
            if not individual.fitness.valid:
                individual[:] = self.prepare(individual)
                individual.fitness.values = (self.evaluate(individual),)
