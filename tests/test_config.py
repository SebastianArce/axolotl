from typing import Any

import pytest

from axolotl.config import SimulationConfig


def test_default_config_is_valid() -> None:
    config = SimulationConfig()
    assert config.steps_per_day == 48
    assert config.n_steps == 28 * 48


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("n_agents", 0),
        ("timestep_minutes", 0),
        ("timestep_minutes", 7),
        ("burn_in_days", 28),
        ("spread", -0.1),
    ],
)
def test_invalid_config_rejected(field: str, value: Any) -> None:
    kwargs: dict[str, Any] = {field: value}
    with pytest.raises(ValueError, match=field):
        SimulationConfig(**kwargs)
