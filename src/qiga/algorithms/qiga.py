"""Quantum-inspired genetic algorithm (QIGA / QEA, Han & Kim 2002) for binary and permutation problems."""

from typing import TYPE_CHECKING
from typing import ClassVar

from qiga.algorithms.base import EvolutionaryAlgorithm
from qiga.config import QIGAConfig
from qiga.operators.rotation import lookup_table_angles
from qiga.operators.rotation import schedule_factor
from qiga.representations.decoders import Decoder
from qiga.representations.decoders import make_decoder
from qiga.representations.qchromosome import QChromosome

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np

    from qiga.problems.base import Problem


class QIGA(EvolutionaryAlgorithm[QIGAConfig]):
    """One generation:

    1. observe every Q-chromosome q_j -> bit string x_j, decode it to a solution
       (repaired if ``config.repair``, and the repair written back into x_j), evaluate;
    2. rotate q_j towards its attractor b_j (lookup table), then optional H_epsilon / NOT gates;
    3. b_j <- x_j if x_j is at least as good;
    4. migration: local (within groups) every generation, global every ``global_migration_period``.

    There is no crossover or selection: information spreads only through the attractors.
    Rotation, attractors and migration all work on bit strings; only evaluation sees the
    decoded solution (identity for binary problems, random keys for permutations).
    """

    name: ClassVar[str] = "qiga"

    def __init__(self, problem: Problem, config: QIGAConfig) -> None:
        super().__init__(problem, config)
        self.decoder: Decoder = make_decoder(problem, config.bits_per_key)
        self.qpopulation: list[QChromosome] = []
        self.observed: list[np.ndarray] = []
        self.observed_fitness: list[float] = []
        self.attractors: list[np.ndarray] = []
        self.attractor_fitness: list[float] = []

    def _initialize(self) -> None:
        self.qpopulation = [QChromosome(self.decoder.n_bits) for _ in range(self.config.population_size)]
        self._observe()
        self.attractors = [bits.copy() for bits in self.observed]
        self.attractor_fitness = list(self.observed_fitness)
        self._migrate()

    def _step(self) -> None:
        self._observe()
        self._rotate()
        self._apply_gates()
        self._update_attractors()
        self._migrate()

    def _population_fitness(self) -> Sequence[float]:
        return self.observed_fitness

    def _observe(self) -> None:
        """Measure every Q-chromosome. Evaluates everything before any state changes, so a
        generation cut short by the evaluation budget leaves the previous state intact."""
        observed, fitness = [], []
        for q in self.qpopulation:
            bits = q.measure(self.rng)
            decoded = self.decoder.decode(bits)
            solution = self.prepare(decoded)
            if solution != decoded:
                bits = self.decoder.encode(solution)
            observed.append(bits)
            fitness.append(self.evaluate(solution))
        self.observed, self.observed_fitness = observed, fitness

    def solution(self, bits: np.ndarray) -> list[int]:
        """Problem solution encoded by a bit string (e.g. an attractor)."""
        return self.decoder.decode(bits)

    def _rotate(self) -> None:
        factor = schedule_factor(
            self._progress(),
            decay=self.config.rotation_schedule == "linear_decay",
            final_ratio=self.config.delta_theta_min / self.config.delta_theta,
        )
        for j, q in enumerate(self.qpopulation):
            angles = lookup_table_angles(
                q.alpha,
                q.beta,
                self.observed[j],
                self.attractors[j],
                observed_at_least_as_good=not self.problem.is_better(self.attractor_fitness[j], self.observed_fitness[j]),
                delta_worse=self.config.delta_theta * factor,
                delta_better=self.config.delta_theta_better * factor,
            )
            q.rotate(angles)

    def _apply_gates(self) -> None:
        for q in self.qpopulation:
            if self.config.not_gate_prob > 0:
                q.apply_not(self.rng.random(q.dimension) < self.config.not_gate_prob)
            if self.config.epsilon > 0:
                q.apply_h_epsilon(self.config.epsilon)

    def _update_attractors(self) -> None:
        for j, (solution, fitness) in enumerate(zip(self.observed, self.observed_fitness, strict=True)):
            if not self.problem.is_better(self.attractor_fitness[j], fitness):
                self.attractors[j] = solution.copy()
                self.attractor_fitness[j] = fitness

    def _migrate(self) -> None:
        period = self.config.global_migration_period
        if period is not None and self.generation > 0 and self.generation % period == 0:
            self._share_best(range(self.config.population_size))
        elif self.config.local_group_size > 1:
            size = self.config.local_group_size
            for start in range(0, self.config.population_size, size):
                self._share_best(range(start, min(start + size, self.config.population_size)))

    def _share_best(self, group: range) -> None:
        best = group[0]
        for j in group:
            if self.problem.is_better(self.attractor_fitness[j], self.attractor_fitness[best]):
                best = j
        for j in group:
            self.attractors[j] = self.attractors[best].copy()
            self.attractor_fitness[j] = self.attractor_fitness[best]

    def _progress(self) -> float:
        if self.config.max_evaluations is not None:
            return self.evaluator.evaluations / self.config.max_evaluations
        return self.generation / self.config.generations
