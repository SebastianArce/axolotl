"""Simulation configuration."""

from dataclasses import dataclass

MINUTES_PER_DAY = 24 * 60


@dataclass(frozen=True)
class SimulationConfig:
    """Parameters of one simulation run.

    Defaults simulate 1,000 agents over 4 weeks at half-hour resolution —
    enough weekend days and plug events for stable time-of-day statistics.
    The first `burn_in_days` are discarded from aggregates so the arbitrary
    initial state of charge does not bias results.
    """

    n_agents: int = 1_000
    n_days: int = 28
    timestep_minutes: int = 30
    burn_in_days: int = 2
    seed: int = 42
    # Scales per-agent variation around archetype means (0 = deterministic).
    spread: float = 1.0

    def __post_init__(self) -> None:
        if self.n_agents <= 0:
            raise ValueError(f"n_agents must be positive, got {self.n_agents}")
        if self.timestep_minutes <= 0 or MINUTES_PER_DAY % self.timestep_minutes:
            raise ValueError(
                f"timestep_minutes must divide a day evenly, got {self.timestep_minutes}"
            )
        if not 0 <= self.burn_in_days < self.n_days:
            raise ValueError(
                f"burn_in_days must be in [0, n_days), got {self.burn_in_days} "
                f"with n_days={self.n_days}"
            )
        if self.spread < 0:
            raise ValueError(f"spread must be non-negative, got {self.spread}")

    @property
    def steps_per_day(self) -> int:
        return MINUTES_PER_DAY // self.timestep_minutes

    @property
    def n_steps(self) -> int:
        return self.n_days * self.steps_per_day
