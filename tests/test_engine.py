import numpy as np
import pytest

from axolotl.archetypes import (
    ARCHETYPES,
    TARGET_SOC_PREFERENCES,
    WEEKEND_TRIPPER,
    Archetype,
)
from axolotl.config import SimulationConfig
from axolotl.engine import run_simulation
from axolotl.prices import CHEAP_WINDOW_END_HOUR, CHEAP_WINDOW_START_HOUR

AVERAGE_UK = ARCHETYPES[0]
INTELLIGENT_OCTOPUS = ARCHETYPES[1]
INFREQUENT_CHARGING = ARCHETYPES[2]
ALWAYS_PLUGGED_IN = ARCHETYPES[5]


def single_archetype_run(archetype: Archetype, **overrides) -> tuple:
    config = SimulationConfig(**{"n_agents": 200, "n_days": 14, **overrides})
    return config, run_simulation(config, archetypes=[archetype])


@pytest.fixture(scope="module")
def population_run() -> tuple:
    config = SimulationConfig(n_agents=300, n_days=14)
    return config, run_simulation(config)


def test_simulation_is_reproducible() -> None:
    config = SimulationConfig(n_agents=50, n_days=7)
    a = run_simulation(config)
    b = run_simulation(config)
    np.testing.assert_array_equal(a.soc, b.soc)
    np.testing.assert_array_equal(a.plugged, b.plugged)


def test_output_shapes_and_bounds(population_run: tuple) -> None:
    config, result = population_run
    assert result.soc.shape == (config.n_agents, config.n_steps)
    assert result.plugged.shape == (config.n_agents, config.n_steps)
    assert np.all(result.soc >= 0)
    assert np.all(result.soc <= 1)


def test_more_plugged_in_overnight_than_midday(population_run: tuple) -> None:
    config, result = population_run
    spd = config.steps_per_day
    after_burn_in = result.plugged[:, config.burn_in_days * spd :]
    by_time_of_day = after_burn_in.reshape(config.n_agents, -1, spd).mean(axis=(0, 1))
    at_3am = by_time_of_day[int(3 / 24 * spd)]
    at_noon = by_time_of_day[int(12 / 24 * spd)]
    assert at_3am > 0.8
    assert at_noon < 0.4


@pytest.mark.parametrize(
    ("index", "tolerance"),
    [(0, 0.05), (1, 0.06), (2, 0.08), (3, 0.05)],
)
def test_plug_in_soc_recapitulates_archetype_table(index: int, tolerance: float) -> None:
    archetype = ARCHETYPES[index]
    config, result = single_archetype_run(archetype)
    keep = result.plug_event_step >= config.burn_in_days * config.steps_per_day
    mean_plug_in_soc = result.plug_event_soc[keep].mean()
    # The archetype table derives plug-in SoC from a flat 0.8 target; agents
    # sample their target from the CNZ preference mix, so the expectation
    # shifts by the difference between the mix's mean and the table's target.
    mean_target = sum(target * weight for target, weight in TARGET_SOC_PREFERENCES)
    expected = archetype.expected_plug_in_soc + (mean_target - archetype.target_soc)
    assert mean_plug_in_soc == pytest.approx(expected, abs=tolerance)


def test_infrequent_chargers_plug_in_rarely() -> None:
    config, result = single_archetype_run(INFREQUENT_CHARGING)
    days_counted = config.n_days - config.burn_in_days
    keep = result.plug_event_step >= config.burn_in_days * config.steps_per_day
    events_per_agent_day = keep.sum() / (config.n_agents * days_counted)
    assert events_per_agent_day == pytest.approx(0.2, abs=0.05)


def test_smart_charging_lands_in_cheap_window() -> None:
    config, result = single_archetype_run(INTELLIGENT_OCTOPUS)
    spd = config.steps_per_day
    charged = np.diff(result.soc, axis=1) > 1e-9
    hours = (np.arange(1, config.n_steps) % spd) * 24 / spd
    in_cheap_window = (hours >= CHEAP_WINDOW_START_HOUR) | (hours < CHEAP_WINDOW_END_HOUR)
    charge_steps = charged.sum()
    assert charge_steps > 0
    assert charged[:, in_cheap_window].sum() / charge_steps > 0.9


def test_smart_charging_follows_each_days_cheap_window() -> None:
    """With a day-by-day series, the schedule must chase each day's cheap slots."""
    config = SimulationConfig(n_agents=50, n_days=7, spread=0.0)
    spd = config.steps_per_day

    def block_hours(day: int) -> tuple[float, float]:
        # The cheap window alternates nightly between 01:00-04:00 and 04:00-07:00.
        return (1.0, 4.0) if day % 2 == 0 else (4.0, 7.0)

    series = []
    for day in range(config.n_days):
        low, high = block_hours(day)
        series.extend(2.0 if low <= step * 24 / spd < high else 30.0 for step in range(spd))

    # Lower mileage keeps the nightly top-up well inside a 3-hour block.
    low_mileage_io = INTELLIGENT_OCTOPUS.model_copy(update={"annual_miles": 9_435})
    result = run_simulation(config, archetypes=[low_mileage_io], price_profile=series)

    charged = np.diff(result.soc, axis=1) > 1e-9  # charge during step s -> diff index s
    steps = np.arange(config.n_steps - 1)
    days, hours = steps // spd, (steps % spd) * 24 / spd
    blocks = np.array([block_hours(day) for day in range(config.n_days)])
    in_todays_block = (hours >= blocks[days][:, 0]) & (hours < blocks[days][:, 1])
    # The final evening's session has its deadline past the simulated horizon,
    # so it has no cheap block to reach — keep it out of the denominator.
    reachable = (days < config.n_days - 1) | (hours < 7.0)
    total_charged_steps = charged[:, reachable].sum()
    assert total_charged_steps > 0
    assert charged[:, in_todays_block].sum() / total_charged_steps > 0.9
    # Both phases of the alternating window see real use: a schedule built from
    # a single typical day could not do this.
    assert charged[:, in_todays_block & (hours < 4.0)].sum() > 0
    assert charged[:, in_todays_block & (hours >= 4.0)].sum() > 0


def test_smart_charging_meets_target_by_departure() -> None:
    config, result = single_archetype_run(INTELLIGENT_OCTOPUS)
    spd = config.steps_per_day
    step_hours = 24 / spd
    # Check SoC at the instant each agent leaves home (recorded before any
    # depletion): charging must be complete by the earlier of the ready-by
    # time and their own departure.
    at_target = []
    for i, agent in enumerate(result.agents):
        for day in range(config.burn_in_days, config.n_days):
            is_weekend = day % 7 in (5, 6)
            hour = agent.weekend_plug_out_hour if is_weekend else agent.plug_out_hour
            plug_out_step = round(hour / step_hours) % spd
            soc_when_leaving = result.soc[i, day * spd + plug_out_step]
            at_target.append(soc_when_leaving >= agent.target_soc - 0.01)
    assert np.mean(at_target) > 0.95


def test_weekend_timing_shifts_show_in_plugged_share() -> None:
    config, result = single_archetype_run(AVERAGE_UK)
    spd = config.steps_per_day
    plugged = result.plugged.reshape(config.n_agents, config.n_days, spd)
    weekdays = [d for d in range(config.burn_in_days, config.n_days) if d % 7 not in (5, 6)]
    weekends = [d for d in range(config.burn_in_days, config.n_days) if d % 7 in (5, 6)]
    at_1730 = int(17.5 / 24 * spd)
    at_0830 = int(8.5 / 24 * spd)
    # Weekend arrivals are ~1h earlier and departures ~2h later, so more of
    # the fleet is plugged in at 17:30 and still plugged in at 08:30.
    assert plugged[:, weekends, at_1730].mean() > plugged[:, weekdays, at_1730].mean() + 0.2
    assert plugged[:, weekends, at_0830].mean() > plugged[:, weekdays, at_0830].mean() + 0.2


def test_always_plugged_in_stays_plugged() -> None:
    _, result = single_archetype_run(ALWAYS_PLUGGED_IN, n_agents=20)
    assert result.plugged.all()


def test_weekend_tripper_depletes_more_at_weekends() -> None:
    config, result = single_archetype_run(WEEKEND_TRIPPER)
    spd = config.steps_per_day
    daily = result.soc.reshape(config.n_agents, config.n_days, spd)
    # Depletion during a day = max SoC that day minus min SoC that day.
    depletion = daily.max(axis=2) - daily.min(axis=2)
    weekday_mean = depletion[:, [7, 8, 9, 10, 11]].mean()
    weekend_mean = depletion[:, [12, 13]].mean()
    assert weekend_mean > 2 * weekday_mean


def test_energy_is_conserved_per_agent() -> None:
    config = SimulationConfig(n_agents=5, n_days=7)
    result = run_simulation(config, archetypes=[AVERAGE_UK])
    for i in range(config.n_agents):
        soc = result.soc[i]
        deltas = np.diff(soc)
        charged = deltas[deltas > 0].sum()
        depleted = -deltas[deltas < 0].sum()
        assert soc[-1] - soc[0] == pytest.approx(charged - depleted, abs=1e-9)


def test_rejects_wrong_price_profile_length() -> None:
    config = SimulationConfig(n_agents=5, n_days=7)
    with pytest.raises(ValueError, match="price_profile"):
        run_simulation(config, price_profile=[10.0] * 3)
