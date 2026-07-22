"""Discrete-time simulation engine.

The engine advances a population of agents through the day in fixed timesteps
with a deliberately explicit loop: at every step each agent may leave home
(start depleting its battery), arrive home (and decide whether to plug in),
and charge according to its strategy. Readability is preferred over
vectorisation so each rule can be read, explained, and changed in isolation.

Daily cycle for one agent:
- At its plug-out time the car leaves; that day's mileage is drawn (gamma
  around the agent's mean, weekday/weekend adjusted) and depleted uniformly
  across the away window.
- At its plug-in time the car arrives home and plugs in if today falls on its
  charging cycle: a frequency of 0.2/day means every 5th day, with a random
  per-agent phase. (A daily Bernoulli draw was considered and rejected: its
  geometric gap lengths regularly run the battery to empty, which both strands
  drivers and biases plug-in SoC well above the archetype table's values.)
- While plugged in, IMMEDIATE agents charge at full power until target SoC;
  SMART agents charge only in the cheapest timesteps that still reach the
  target by their ready-by deadline (Intelligent Octopus-style automation).
"""

import math
from dataclasses import dataclass

import numpy as np

from axolotl.agent import Agent, sample_population
from axolotl.archetypes import ARCHETYPES, Archetype, ChargingStrategy
from axolotl.config import SimulationConfig
from axolotl.prices import synthetic_price_profile

# Shape of the gamma distribution for day-to-day mileage variation. Shape 4
# gives a right-skewed distribution with a coefficient of variation of 0.5:
# many ordinary days, occasional long trips, never negative.
DAILY_MILES_GAMMA_SHAPE = 4.0

DAYS_PER_WEEK = 7
WEEKEND_DAY_INDICES = (5, 6)  # simulation starts on a Monday


@dataclass
class SimulationResult:
    """Raw simulation output: one row per agent, one column per timestep."""

    soc: np.ndarray  # float, shape (n_agents, n_steps)
    plugged: np.ndarray  # bool, shape (n_agents, n_steps)
    # One entry per plug event: which agent, at which step, at what SoC.
    plug_event_agent: np.ndarray
    plug_event_step: np.ndarray
    plug_event_soc: np.ndarray
    agents: list[Agent]
    config: SimulationConfig


class _AgentState:
    """Mutable per-agent state while the simulation runs."""

    __slots__ = (
        "agent",
        "away_steps",
        "charge_schedule",
        "depletion_per_step",
        "is_away",
        "plug_in_step",
        "plug_interval_days",
        "plug_out_step",
        "plug_phase",
        "plugged",
        "soc",
    )

    def __init__(self, agent: Agent, steps_per_day: int, rng: np.random.Generator) -> None:
        self.agent = agent
        step_hours = 24 / steps_per_day
        self.plug_in_step = round(agent.plug_in_hour / step_hours) % steps_per_day
        self.plug_out_step = round(agent.plug_out_hour / step_hours) % steps_per_day
        self.away_steps = (self.plug_in_step - self.plug_out_step) % steps_per_day
        # Plug in every k-th day (frequency 0.2 -> every 5 days); random phase
        # so agents on the same cycle are not synchronised.
        self.plug_interval_days = max(1, round(1 / agent.archetype.plug_in_frequency_per_day))
        self.plug_phase = int(rng.integers(self.plug_interval_days))
        # Start at midnight of day 0: home, plugged in at target SoC. The
        # burn-in period lets each agent settle into its own rhythm.
        self.soc = agent.target_soc
        self.plugged = True
        self.is_away = False
        self.depletion_per_step = 0.0
        self.charge_schedule: set[int] = set()


def run_simulation(
    config: SimulationConfig,
    archetypes: list[Archetype] | None = None,
    price_profile: list[float] | None = None,
) -> SimulationResult:
    """Simulate a population of EV drivers and record plug-in state and SoC."""
    steps_per_day = config.steps_per_day
    prices = price_profile if price_profile is not None else synthetic_price_profile(steps_per_day)
    if len(prices) != steps_per_day:
        raise ValueError(f"price_profile must have {steps_per_day} entries, got {len(prices)}")

    rng = np.random.default_rng(config.seed)
    agents = sample_population(
        archetypes if archetypes is not None else list(ARCHETYPES),
        config.n_agents,
        rng,
        config.spread,
    )
    states = [_AgentState(agent, steps_per_day, rng) for agent in agents]

    step_hours = 24 / steps_per_day
    soc = np.zeros((len(agents), config.n_steps))
    plugged = np.zeros((len(agents), config.n_steps), dtype=bool)
    events_agent: list[int] = []
    events_step: list[int] = []
    events_soc: list[float] = []

    for step in range(config.n_steps):
        step_of_day = step % steps_per_day
        day = step // steps_per_day
        is_weekend = day % DAYS_PER_WEEK in WEEKEND_DAY_INDICES

        for i, state in enumerate(states):
            archetype = state.agent.archetype

            if step_of_day == state.plug_out_step and state.away_steps > 0:
                _leave_home(state, rng, is_weekend)

            if step_of_day == state.plug_in_step:
                state.is_away = False
                if (day - state.plug_phase) % state.plug_interval_days == 0:
                    state.plugged = True
                    events_agent.append(i)
                    events_step.append(step)
                    events_soc.append(state.soc)
                    if archetype.strategy is ChargingStrategy.SMART:
                        state.charge_schedule = _smart_schedule(
                            state, step, steps_per_day, prices, step_hours
                        )

            if state.is_away:
                state.soc = max(state.soc - state.depletion_per_step, 0.0)
            elif (
                state.plugged
                and state.soc < state.agent.target_soc
                and (
                    archetype.strategy is ChargingStrategy.IMMEDIATE
                    or step in state.charge_schedule
                )
            ):
                _charge(state, step_hours)

            soc[i, step] = state.soc
            plugged[i, step] = state.plugged

    return SimulationResult(
        soc=soc,
        plugged=plugged,
        plug_event_agent=np.array(events_agent),
        plug_event_step=np.array(events_step),
        plug_event_soc=np.array(events_soc),
        agents=agents,
        config=config,
    )


def _leave_home(state: _AgentState, rng: np.random.Generator, is_weekend: bool) -> None:
    """Unplug and start driving: draw today's miles, deplete evenly while away."""
    agent = state.agent
    state.plugged = False
    state.is_away = True
    state.charge_schedule = set()

    mean_miles = agent.mean_daily_miles * agent.archetype.daily_miles_multiplier(is_weekend)
    miles = rng.gamma(DAILY_MILES_GAMMA_SHAPE, mean_miles / DAILY_MILES_GAMMA_SHAPE)
    kwh_needed = miles / agent.archetype.efficiency_mi_per_kwh
    state.depletion_per_step = kwh_needed / agent.archetype.battery_kwh / state.away_steps


def _smart_schedule(
    state: _AgentState,
    plug_step: int,
    steps_per_day: int,
    prices: list[float],
    step_hours: float,
) -> set[int]:
    """Pick the cheapest timesteps that reach target SoC by the ready-by deadline.

    Mirrors the Intelligent Octopus automation described in the CNZ report:
    on plug-in, compute how long charging will take, then schedule it into the
    cheapest slots before the deadline. The effective deadline is the ready-by
    time or the driver's own departure, whichever comes first — a driver who
    leaves at 6:30 needs the car ready then, not at 7:00.
    """
    agent = state.agent
    archetype = agent.archetype
    kwh_needed = (agent.target_soc - state.soc) * archetype.battery_kwh
    if kwh_needed <= 0:
        return set()
    steps_needed = math.ceil(kwh_needed / (archetype.charger_kw * step_hours))

    ready_by_step = round(archetype.ready_by_hour / step_hours) % steps_per_day
    deadline = min(
        _next_occurrence(ready_by_step, plug_step, steps_per_day),
        _next_occurrence(state.plug_out_step, plug_step, steps_per_day),
    )

    candidates = range(plug_step, deadline)
    cheapest = sorted(candidates, key=lambda s: (prices[s % steps_per_day], s))
    return set(cheapest[:steps_needed])


def _next_occurrence(step_of_day: int, after_step: int, steps_per_day: int) -> int:
    """First absolute step strictly after `after_step` that falls on `step_of_day`."""
    step = (after_step // steps_per_day) * steps_per_day + step_of_day
    while step <= after_step:
        step += steps_per_day
    return step


def _charge(state: _AgentState, step_hours: float) -> None:
    """Add one timestep of charge, never exceeding the target SoC."""
    archetype = state.agent.archetype
    max_kwh = archetype.charger_kw * step_hours
    headroom_kwh = (state.agent.target_soc - state.soc) * archetype.battery_kwh
    state.soc += min(max_kwh, headroom_kwh) / archetype.battery_kwh
