import pytest

from axolotl.aggregate import plug_in_soc_stats, time_of_day_profile
from axolotl.config import SimulationConfig
from axolotl.engine import SimulationResult, run_simulation


@pytest.fixture(scope="module")
def result() -> SimulationResult:
    return run_simulation(SimulationConfig(n_agents=300, n_days=14))


def test_profile_has_one_row_per_timestep(result: SimulationResult) -> None:
    profile = time_of_day_profile(result)
    assert profile.height == result.config.steps_per_day
    assert profile["hour"].to_list()[0] == 0.0
    assert (profile["pct_plugged_in"] >= 0).all()
    assert (profile["pct_plugged_in"] <= 100).all()


def test_profile_percentiles_are_ordered(result: SimulationResult) -> None:
    profile = time_of_day_profile(result)
    assert (profile["soc_p05"] <= profile["soc_p25"]).all()
    assert (profile["soc_p25"] <= profile["soc_p75"]).all()
    assert (profile["soc_p75"] <= profile["soc_p95"]).all()
    assert (profile["soc_p05"] <= profile["soc_mean"]).all()
    assert (profile["soc_mean"] <= profile["soc_p95"]).all()


def test_weekday_and_weekend_filters_differ(result: SimulationResult) -> None:
    weekday = time_of_day_profile(result, days="weekday")
    weekend = time_of_day_profile(result, days="weekend")
    assert weekday.height == weekend.height
    assert weekday["pct_plugged_in"].to_list() != weekend["pct_plugged_in"].to_list()


def test_plug_in_soc_stats_covers_all_archetypes(result: SimulationResult) -> None:
    stats = plug_in_soc_stats(result)
    assert stats.height == len({a.archetype.name for a in result.agents})
    assert (stats["n_events"] > 0).all()
    assert (stats["mean"] >= 0).all()
    assert (stats["mean"] <= 1).all()
