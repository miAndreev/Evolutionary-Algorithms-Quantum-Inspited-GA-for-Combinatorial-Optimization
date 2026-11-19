import pydantic
import pytest

from qiga.config import GAConfig


def test_defaults_are_valid():
    config = GAConfig()

    assert config.population_size == 50
    assert config.elitism < config.population_size


@pytest.mark.parametrize(
    "kwargs",
    [
        {"population_size": 10, "elitism": 10},
        {"population_size": 1},
        {"generations": 0},
        {"max_evaluations": 0},
        {"elitism": -1},
        {"seed": -5},
        {"unknown_option": 1},
    ],
)
def test_invalid_config_raises_value_error(kwargs):
    with pytest.raises(ValueError):
        GAConfig(**kwargs)


def test_config_is_immutable():
    config = GAConfig()

    with pytest.raises(pydantic.ValidationError):
        config.population_size = 10


def test_config_round_trips_through_json():
    config = GAConfig(population_size=20, generations=5, max_evaluations=1000, target_fitness=42.0, seed=7)

    assert GAConfig.model_validate_json(config.model_dump_json()) == config
