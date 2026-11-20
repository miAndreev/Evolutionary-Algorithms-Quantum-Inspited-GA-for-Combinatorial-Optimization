"""Name → DEAP operator lookup, checked against the problem's encoding.

All operators work in place on list-like individuals and draw randomness from Python's
global ``random`` module (a DEAP convention), which the algorithm seeds.
"""

from collections.abc import Callable
from functools import partial
import random
from typing import Any

from deap import tools

from qiga.problems.base import Encoding
from qiga.problems.base import Sense

type Crossover = Callable[[Any, Any], tuple[Any, Any]]
type Mutation = Callable[[Any], tuple[Any]]
type Selection = Callable[[list[Any], int], list[Any]]


def mut_inversion(individual: list[int]) -> tuple[list[int]]:
    """Reverse a random segment (a 2-opt move for tours); keeps the permutation valid."""
    i, j = sorted(random.sample(range(len(individual)), 2))
    individual[i : j + 1] = individual[i : j + 1][::-1]
    return (individual,)


DEFAULT_CROSSOVER: dict[Encoding, str] = {Encoding.BINARY: "two_point", Encoding.PERMUTATION: "ordered"}
DEFAULT_MUTATION: dict[Encoding, str] = {Encoding.BINARY: "flip_bit", Encoding.PERMUTATION: "inversion"}

_CROSSOVERS: dict[Encoding, dict[str, Crossover]] = {
    Encoding.BINARY: {
        "one_point": tools.cxOnePoint,
        "two_point": tools.cxTwoPoint,
        "uniform": partial(tools.cxUniform, indpb=0.5),
    },
    Encoding.PERMUTATION: {
        "ordered": tools.cxOrdered,
        "pmx": tools.cxPartialyMatched,
    },
}

_MUTATIONS: dict[Encoding, dict[str, Callable[[float], Mutation]]] = {
    Encoding.BINARY: {
        "flip_bit": lambda gene_prob: partial(tools.mutFlipBit, indpb=gene_prob),
    },
    Encoding.PERMUTATION: {
        "shuffle": lambda gene_prob: partial(tools.mutShuffleIndexes, indpb=gene_prob),
        "inversion": lambda _: mut_inversion,
    },
}


def make_crossover(name: str | None, encoding: Encoding) -> Crossover:
    name = name or DEFAULT_CROSSOVER[encoding]
    return _lookup(_CROSSOVERS, "crossover", name, encoding)


def make_mutation(name: str | None, encoding: Encoding, gene_prob: float) -> Mutation:
    name = name or DEFAULT_MUTATION[encoding]
    return _lookup(_MUTATIONS, "mutation", name, encoding)(gene_prob)


def make_selection(name: str, sense: Sense, tournament_size: int) -> Selection:
    if name == "tournament":
        return partial(tools.selTournament, tournsize=tournament_size)
    if name == "roulette":
        if sense is Sense.MINIMIZE:
            msg = "roulette selection requires a maximization problem with non-negative fitness"
            raise ValueError(msg)
        return tools.selRoulette
    msg = f"unknown selection {name!r}"
    raise ValueError(msg)


def _lookup[T](table: dict[Encoding, dict[str, T]], kind: str, name: str, encoding: Encoding) -> T:
    try:
        return table[encoding][name]
    except KeyError:
        available = ", ".join(sorted(table[encoding]))
        msg = f"{kind} {name!r} is not available for {encoding} encoding (available: {available})"
        raise ValueError(msg) from None
