import numpy as np
import pytest

from axolotl.agent import sample_agent, sample_population
from axolotl.archetypes import ARCHETYPES

AVERAGE_UK = ARCHETYPES[0]


def test_same_seed_gives_identical_agents() -> None:
    a = sample_agent(AVERAGE_UK, np.random.default_rng(7))
    b = sample_agent(AVERAGE_UK, np.random.default_rng(7))
    assert a == b


def test_zero_spread_reproduces_archetype_means() -> None:
    agent = sample_agent(AVERAGE_UK, np.random.default_rng(0), spread=0.0)
    assert agent.plug_in_hour == pytest.approx(AVERAGE_UK.plug_in_hour)
    assert agent.plug_out_hour == pytest.approx(AVERAGE_UK.plug_out_hour)
    assert agent.mean_daily_miles == pytest.approx(AVERAGE_UK.mean_daily_miles)


def test_population_recovers_archetype_means() -> None:
    rng = np.random.default_rng(1)
    agents = [sample_agent(AVERAGE_UK, rng) for _ in range(5_000)]
    mean_miles = np.mean([a.mean_daily_miles for a in agents])
    mean_plug_in = np.mean([a.plug_in_hour for a in agents])
    assert mean_miles == pytest.approx(AVERAGE_UK.mean_daily_miles, rel=0.02)
    assert mean_plug_in == pytest.approx(AVERAGE_UK.plug_in_hour, abs=0.05)


def test_sampled_hours_stay_in_day_range() -> None:
    rng = np.random.default_rng(2)
    for _ in range(1_000):
        agent = sample_agent(AVERAGE_UK, rng)
        assert 0 <= agent.plug_in_hour < 24
        assert 0 <= agent.plug_out_hour < 24
        assert agent.mean_daily_miles > 0


def test_population_counts_are_proportional_and_exact() -> None:
    agents = sample_population(list(ARCHETYPES), 1_000, np.random.default_rng(3))
    assert len(agents) == 1_000
    counts = {a.name: 0 for a in ARCHETYPES}
    for agent in agents:
        counts[agent.archetype.name] += 1
    for archetype in ARCHETYPES:
        assert counts[archetype.name] == round(archetype.population_share * 1_000)


def test_population_renormalises_archetype_subset() -> None:
    subset = list(ARCHETYPES[:2])  # shares 0.40 and 0.30
    agents = sample_population(subset, 700, np.random.default_rng(4))
    assert len(agents) == 700
    n_average = sum(a.archetype is subset[0] for a in agents)
    assert n_average == 400


def test_population_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="n_agents"):
        sample_population(list(ARCHETYPES), 0, np.random.default_rng(0))
