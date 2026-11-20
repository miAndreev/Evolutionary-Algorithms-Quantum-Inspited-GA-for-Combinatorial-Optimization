"""Algorithm name → (implementation, config model). New algorithms register here."""

from typing import TYPE_CHECKING

from qiga.algorithms.classical_ga import ClassicalGA
from qiga.algorithms.qiga import QIGA
from qiga.algorithms.random_search import RandomSearch
from qiga.config import ClassicalGAConfig
from qiga.config import GAConfig
from qiga.config import QIGAConfig

if TYPE_CHECKING:
    from qiga.algorithms.base import EvolutionaryAlgorithm

ALGORITHMS: dict[str, tuple[type[EvolutionaryAlgorithm], type[GAConfig]]] = {
    "classical_ga": (ClassicalGA, ClassicalGAConfig),
    "qiga": (QIGA, QIGAConfig),
    "random_search": (RandomSearch, GAConfig),
}
