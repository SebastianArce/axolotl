"""Simulation configuration."""

from pydantic import BaseModel, ConfigDict, Field, model_validator

MINUTES_PER_DAY = 24 * 60


class SimulationConfig(BaseModel):
    """Parameters of one simulation run.

    Defaults simulate 1,000 agents over 4 weeks at half-hour resolution —
    enough weekend days and plug events for stable time-of-day statistics.
    The first `burn_in_days` are discarded from aggregates so the arbitrary
    initial state of charge does not bias results.
    """

    model_config = ConfigDict(frozen=True)

    n_agents: int = Field(default=1_000, gt=0)
    n_days: int = Field(default=28, gt=0)
    timestep_minutes: int = Field(default=30, gt=0)
    burn_in_days: int = Field(default=2, ge=0)
    seed: int = 42
    # Scales per-agent variation around archetype means (0 = deterministic).
    spread: float = Field(default=1.0, ge=0)

    @model_validator(mode="after")
    def _validate_cross_field(self) -> "SimulationConfig":
        if MINUTES_PER_DAY % self.timestep_minutes:
            raise ValueError(
                f"timestep_minutes must divide a day evenly, got {self.timestep_minutes}"
            )
        if self.burn_in_days >= self.n_days:
            raise ValueError(
                f"burn_in_days must be less than n_days, got {self.burn_in_days} "
                f"with n_days={self.n_days}"
            )
        return self

    @property
    def steps_per_day(self) -> int:
        return MINUTES_PER_DAY // self.timestep_minutes

    @property
    def n_steps(self) -> int:
        return self.n_days * self.steps_per_day
