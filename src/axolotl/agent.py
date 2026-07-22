"""Agents: individual drivers sampled from an archetype.

An agent is an immutable set of behavioural parameters. Day-to-day randomness
(whether to plug in, that day's mileage) is drawn by the simulation engine;
this module only captures *who the driver is*, not what they do on a given day.
"""

import numpy as np
from pydantic import BaseModel, ConfigDict

from axolotl.archetypes import TARGET_SOC_PREFERENCES, Archetype

# Weekend times get extra per-agent jitter on top of the weekday habit —
# weekend behaviour is visibly more spread out in CNZ report Fig. 4. Scales
# the archetype's plug_time_sigma_hours.
WEEKEND_EXTRA_TIME_SIGMA = 1.0


class Agent(BaseModel):
    """One driver, sampled from an archetype with individual variation."""

    model_config = ConfigDict(frozen=True)

    archetype: Archetype
    # This agent's habitual plug-in/out times, fractional local hours.
    plug_in_hour: float
    plug_out_hour: float
    # Weekend habits: the weekday times shifted (earlier arrival, later
    # departure) with extra jitter.
    weekend_plug_in_hour: float
    weekend_plug_out_hour: float
    # This agent's mean daily mileage (day-to-day variation applied by the engine).
    mean_daily_miles: float
    # This agent's charging-target preference (sampled from CNZ Fig. 2).
    target_soc: float


def sample_agent(archetype: Archetype, rng: np.random.Generator, spread: float = 1.0) -> Agent:
    """Draw one agent from an archetype.

    `spread` scales all per-agent variation: 0 reproduces the archetype's
    parameters exactly (including its flat charging target), 1 uses the
    archetype's default sigmas and the report's target-preference mix.
    """
    time_sigma = archetype.plug_time_sigma_hours * spread
    plug_in = rng.normal(archetype.plug_in_hour, time_sigma) % 24
    plug_out = rng.normal(archetype.plug_out_hour, time_sigma) % 24

    weekend_sigma = archetype.plug_time_sigma_hours * WEEKEND_EXTRA_TIME_SIGMA * spread
    weekend_plug_in = rng.normal(plug_in + archetype.weekend_plug_in_shift_hours, weekend_sigma)
    weekend_plug_out = rng.normal(plug_out + archetype.weekend_plug_out_shift_hours, weekend_sigma)

    # Lognormal with unit mean so the population average recovers the archetype mean.
    miles_sigma = archetype.miles_sigma * spread
    miles_factor = rng.lognormal(mean=-(miles_sigma**2) / 2, sigma=miles_sigma)

    if spread == 0:
        target_soc = archetype.target_soc
    else:
        targets, weights = zip(*TARGET_SOC_PREFERENCES, strict=True)
        target_soc = float(rng.choice(targets, p=weights))

    return Agent(
        archetype=archetype,
        plug_in_hour=plug_in,
        plug_out_hour=plug_out,
        weekend_plug_in_hour=weekend_plug_in % 24,
        weekend_plug_out_hour=weekend_plug_out % 24,
        mean_daily_miles=archetype.mean_daily_miles * miles_factor,
        target_soc=target_soc,
    )


def sample_population(
    archetypes: list[Archetype],
    n_agents: int,
    rng: np.random.Generator,
    spread: float = 1.0,
) -> list[Agent]:
    """Sample a population with archetype counts proportional to population share.

    Shares are renormalised over the given archetypes, so a subset (e.g. a
    dashboard multi-select) still yields a full population. If all selected
    shares are zero (e.g. illustrative presets), agents split evenly. Counts
    use largest-remainder rounding so they always sum to `n_agents` exactly.
    """
    if n_agents <= 0:
        raise ValueError(f"n_agents must be positive, got {n_agents}")
    if not archetypes:
        raise ValueError("archetypes must not be empty")
    total_share = sum(a.population_share for a in archetypes)
    shares = (
        [a.population_share / total_share for a in archetypes]
        if total_share > 0
        else [1 / len(archetypes)] * len(archetypes)
    )

    quotas = [share * n_agents for share in shares]
    counts = [int(q) for q in quotas]
    remainders = [q - c for q, c in zip(quotas, counts, strict=True)]
    for i in sorted(range(len(quotas)), key=lambda i: remainders[i], reverse=True)[
        : n_agents - sum(counts)
    ]:
        counts[i] += 1

    return [
        sample_agent(archetype, rng, spread)
        for archetype, count in zip(archetypes, counts, strict=True)
        for _ in range(count)
    ]
