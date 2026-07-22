"""Agents: individual drivers sampled from an archetype.

An agent is an immutable set of behavioural parameters. Day-to-day randomness
(whether to plug in, that day's mileage) is drawn by the simulation engine;
this module only captures *who the driver is*, not what they do on a given day.
"""

from dataclasses import dataclass

import numpy as np

from axolotl.archetypes import Archetype


@dataclass(frozen=True)
class Agent:
    """One driver, sampled from an archetype with individual variation."""

    archetype: Archetype
    # This agent's habitual plug-in/out times, fractional local hours.
    plug_in_hour: float
    plug_out_hour: float
    # This agent's mean daily mileage (day-to-day variation applied by the engine).
    mean_daily_miles: float


def sample_agent(archetype: Archetype, rng: np.random.Generator, spread: float = 1.0) -> Agent:
    """Draw one agent from an archetype.

    `spread` scales all per-agent variation: 0 reproduces the archetype means
    exactly, 1 uses the archetype's default sigmas.
    """
    time_sigma = archetype.plug_time_sigma_hours * spread
    plug_in = rng.normal(archetype.plug_in_hour, time_sigma) % 24
    plug_out = rng.normal(archetype.plug_out_hour, time_sigma) % 24

    # Lognormal with unit mean so the population average recovers the archetype mean.
    miles_sigma = archetype.miles_sigma * spread
    miles_factor = rng.lognormal(mean=-(miles_sigma**2) / 2, sigma=miles_sigma)

    return Agent(
        archetype=archetype,
        plug_in_hour=plug_in,
        plug_out_hour=plug_out,
        mean_daily_miles=archetype.mean_daily_miles * miles_factor,
    )


def sample_population(
    archetypes: list[Archetype],
    n_agents: int,
    rng: np.random.Generator,
    spread: float = 1.0,
) -> list[Agent]:
    """Sample a population with archetype counts proportional to population share.

    Shares are renormalised over the given archetypes, so a subset (e.g. a
    dashboard multi-select) still yields a full population. Counts use
    largest-remainder rounding so they always sum to `n_agents` exactly.
    """
    if n_agents <= 0:
        raise ValueError(f"n_agents must be positive, got {n_agents}")
    total_share = sum(a.population_share for a in archetypes)
    if total_share <= 0:
        raise ValueError("selected archetypes have zero total population share")

    quotas = [a.population_share / total_share * n_agents for a in archetypes]
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
