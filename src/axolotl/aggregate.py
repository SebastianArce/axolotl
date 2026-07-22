"""Population-level aggregation of raw simulation output.

Produces the time-of-day profile behind the dashboard chart: for each timestep
of the day, the share of the population plugged in and the distribution of
state of charge (mean and percentile band), pooled over agents and days.
"""

from typing import Literal

import numpy as np
import polars as pl

from axolotl.engine import SimulationResult

DAYS_PER_WEEK = 7
WEEKEND_DAY_INDICES = (5, 6)

DayFilter = Literal["all", "weekday", "weekend"]


def time_of_day_profile(result: SimulationResult, days: DayFilter = "all") -> pl.DataFrame:
    """Aggregate a simulation into a per-timestep-of-day population profile.

    Burn-in days are excluded. Returns one row per timestep of the day with:
    hour, pct_plugged_in, and SoC mean / 5th / 25th / 75th / 95th percentiles
    (all SoC values in percent).
    """
    config = result.config
    spd = config.steps_per_day
    n_agents = result.soc.shape[0]

    kept_days = [
        day
        for day in range(config.burn_in_days, config.n_days)
        if days == "all" or (days == "weekend") == (day % DAYS_PER_WEEK in WEEKEND_DAY_INDICES)
    ]
    if not kept_days:
        raise ValueError(f"no days match filter {days!r} after burn-in")

    # (n_agents, n_days, steps_per_day) so axis 2 is time of day.
    soc = result.soc.reshape(n_agents, config.n_days, spd)[:, kept_days, :] * 100
    plugged = result.plugged.reshape(n_agents, config.n_days, spd)[:, kept_days, :]

    p05, p25, p75, p95 = np.percentile(soc, [5, 25, 75, 95], axis=(0, 1))
    return pl.DataFrame(
        {
            "hour": [step * 24 / spd for step in range(spd)],
            "pct_plugged_in": plugged.mean(axis=(0, 1)) * 100,
            "soc_mean": soc.mean(axis=(0, 1)),
            "soc_p05": p05,
            "soc_p25": p25,
            "soc_p75": p75,
            "soc_p95": p95,
        }
    )


def plug_in_soc_stats(result: SimulationResult) -> pl.DataFrame:
    """Per-archetype summary of SoC at plug-in, excluding burn-in days.

    Used to check the simulation recapitulates population-level observations
    (the archetype table's plug-in SoC column; CNZ report Figure 7).
    """
    config = result.config
    first_step = config.burn_in_days * config.steps_per_day
    keep = result.plug_event_step >= first_step

    names = np.array([result.agents[i].archetype.name for i in result.plug_event_agent[keep]])
    soc = result.plug_event_soc[keep]
    return (
        pl.DataFrame({"archetype": names, "plug_in_soc": soc})
        .group_by("archetype")
        .agg(
            pl.col("plug_in_soc").mean().alias("mean"),
            pl.col("plug_in_soc").median().alias("median"),
            pl.len().alias("n_events"),
        )
    )
