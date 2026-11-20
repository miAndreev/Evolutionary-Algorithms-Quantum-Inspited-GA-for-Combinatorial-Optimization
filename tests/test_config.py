import math

import pydantic
import pytest

from qiga.config import ClassicalGAConfig
from qiga.config import GAConfig
from qiga.config import QIGAConfig


def test_defaults_are_valid():
    assert GAConfig().population_size == 50
    assert ClassicalGAConfig().elitism < ClassicalGAConfig().population_size
    assert QIGAConfig().population_size == 10


@pytest.mark.parametrize(
    "kwargs",
    [
        {"population_size": 1},
        {"generations": 0},
        {"max_evaluations": 0},
        {"population_size": 10, "max_evaluations": 5},
        {"seed": -5},
        {"unknown_option": 1},
        {"elitism": 1},
    ],
)
def test_invalid_config_raises_value_error(kwargs):
    with pytest.raises(ValueError):
        GAConfig(**kwargs)


@pytest.mark.parametrize("kwargs", [{"population_size": 10, "elitism": 10}, {"elitism": -1}, {"tournament_size": 1}, {"crossover": "bogus"}])
def test_invalid_classical_config(kwargs):
    with pytest.raises(ValueError):
        ClassicalGAConfig(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"delta_theta": 0},
        {"delta_theta": math.pi},
        {"delta_theta": 0.01, "delta_theta_min": 0.02},
        {"population_size": 4, "local_group_size": 5},
        {"epsilon": 0.5},
        {"not_gate_prob": 1.5},
        {"rotation_schedule": "exponential"},
        {"elitism": 1},
    ],
)
def test_invalid_qiga_config(kwargs):
    with pytest.raises(ValueError):
        QIGAConfig(**kwargs)


def test_config_is_immutable():
    config = GAConfig()

    with pytest.raises(pydantic.ValidationError):
        config.population_size = 10


@pytest.mark.parametrize("config_cls", [GAConfig, ClassicalGAConfig, QIGAConfig])
def test_config_round_trips_through_json(config_cls):
    config = config_cls(population_size=20, generations=5, max_evaluations=1000, target_fitness=42.0, seed=7)

    assert config_cls.model_validate_json(config.model_dump_json()) == config
