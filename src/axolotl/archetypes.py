"""Driver archetypes: population-level presets the simulation samples agents from.

The six presets transcribe the archetype table provided with the task, itself
derived from Centre for Net Zero's "Learning from Intelligent Octopus" report
(May 2022). Times are local hours; SoC values are fractions of battery capacity.
"""

from dataclasses import dataclass
from enum import StrEnum

DAYS_PER_YEAR = 365
WEEKDAYS_PER_WEEK = 5
WEEKEND_DAYS_PER_WEEK = 2


class ChargingStrategy(StrEnum):
    """How the car charges while plugged in."""

    # Charge at full charger power from plug-in until target SoC is reached.
    IMMEDIATE = "immediate"
    # Charge in the cheapest half-hour slots so target SoC is reached by
    # `ready_by_hour` (Intelligent Octopus-style automation).
    SMART = "smart"


@dataclass(frozen=True)
class Archetype:
    """Parameters describing one behavioural pattern of EV drivers.

    Per-agent variation is applied on top of these means when sampling a
    population (see `agent.sample_population`).
    """

    name: str
    population_share: float
    annual_miles: float
    battery_kwh: float
    efficiency_mi_per_kwh: float
    # Probability of plugging in on any given day (1.0 = daily, 0.2 = every ~5 days).
    plug_in_frequency_per_day: float
    charger_kw: float
    # Mean plug-in/plug-out times, fractional local hours (18.5 = 18:30).
    plug_in_hour: float
    plug_out_hour: float
    target_soc: float
    strategy: ChargingStrategy = ChargingStrategy.IMMEDIATE
    # SMART only: deadline by which target SoC should be reached.
    ready_by_hour: float = 7.0
    # Std dev of per-agent plug-in/out times. ~1h spread around the evening
    # peak per CNZ report Fig. 4; 0 disables time variation.
    plug_time_sigma_hours: float = 1.0
    # Lognormal sigma of per-agent mean daily mileage around the archetype mean.
    miles_sigma: float = 0.25
    # Weekend daily miles relative to the archetype's average day. Weekday
    # miles are scaled so annual mileage is preserved (see multiplier methods).
    weekend_miles_multiplier: float = 1.0

    def __post_init__(self) -> None:
        if not 0 <= self.population_share <= 1:
            raise ValueError(f"population_share must be in [0, 1], got {self.population_share}")
        if not 0 < self.target_soc <= 1:
            raise ValueError(f"target_soc must be in (0, 1], got {self.target_soc}")
        if not 0 < self.plug_in_frequency_per_day <= 1:
            raise ValueError(
                f"plug_in_frequency_per_day must be in (0, 1], got {self.plug_in_frequency_per_day}"
            )
        max_multiplier = 7 / WEEKEND_DAYS_PER_WEEK
        if not 0 <= self.weekend_miles_multiplier <= max_multiplier:
            raise ValueError(
                f"weekend_miles_multiplier must be in [0, {max_multiplier}], "
                f"got {self.weekend_miles_multiplier}"
            )

    # -- Derived quantities (the spreadsheet's computed columns) ---------------

    @property
    def mean_daily_miles(self) -> float:
        return self.annual_miles / DAYS_PER_YEAR

    @property
    def kwh_per_year(self) -> float:
        return self.annual_miles / self.efficiency_mi_per_kwh

    @property
    def kwh_per_plug_in(self) -> float:
        """Average energy needed per plug event to cover driving between plug-ins."""
        return self.kwh_per_year / DAYS_PER_YEAR / self.plug_in_frequency_per_day

    @property
    def expected_plug_in_soc(self) -> float:
        """SoC expected at plug-in if the car left the last session at target SoC."""
        return self.target_soc - self.kwh_per_plug_in / self.battery_kwh

    @property
    def expected_charging_hours(self) -> float:
        """Hours of charging needed per plug event at full charger power."""
        return self.kwh_per_plug_in / self.charger_kw

    def daily_miles_multiplier(self, is_weekend: bool) -> float:
        """Scale factor on mean daily miles for a weekend or weekday day.

        Weekday scaling is chosen so that total weekly (and hence annual)
        mileage is unchanged: (5 * weekday + 2 * weekend) / 7 == 1.
        """
        if is_weekend:
            return self.weekend_miles_multiplier
        return (7 - WEEKEND_DAYS_PER_WEEK * self.weekend_miles_multiplier) / WEEKDAYS_PER_WEEK


ARCHETYPES: tuple[Archetype, ...] = (
    Archetype(
        name="Average (UK)",
        population_share=0.40,
        annual_miles=9_435,
        battery_kwh=60.0,
        efficiency_mi_per_kwh=3.5,
        plug_in_frequency_per_day=1.0,
        charger_kw=7.0,
        plug_in_hour=18.0,
        plug_out_hour=7.0,
        target_soc=0.80,
    ),
    Archetype(
        name="Intelligent Octopus average",
        population_share=0.30,
        annual_miles=28_105,
        battery_kwh=72.5,
        efficiency_mi_per_kwh=3.5,
        plug_in_frequency_per_day=1.0,
        charger_kw=7.0,
        plug_in_hour=18.0,
        plug_out_hour=7.0,
        target_soc=0.80,
        strategy=ChargingStrategy.SMART,
        ready_by_hour=7.0,
    ),
    Archetype(
        name="Infrequent charging",
        population_share=0.10,
        annual_miles=9_435,
        battery_kwh=60.0,
        efficiency_mi_per_kwh=3.5,
        plug_in_frequency_per_day=0.2,
        charger_kw=7.0,
        plug_in_hour=18.0,
        plug_out_hour=7.0,
        target_soc=0.80,
    ),
    Archetype(
        name="Infrequent driving",
        population_share=0.10,
        annual_miles=5_700,
        battery_kwh=60.0,
        efficiency_mi_per_kwh=3.5,
        plug_in_frequency_per_day=1.0,
        charger_kw=7.0,
        plug_in_hour=18.0,
        plug_out_hour=7.0,
        target_soc=0.80,
    ),
    Archetype(
        name="Scheduled charging",
        population_share=0.09,
        annual_miles=9_435,
        battery_kwh=60.0,
        efficiency_mi_per_kwh=3.5,
        plug_in_frequency_per_day=1.0,
        charger_kw=7.0,
        plug_in_hour=22.0,
        plug_out_hour=9.0,
        target_soc=0.80,
    ),
    Archetype(
        name="Always plugged-in",
        population_share=0.01,
        annual_miles=9_435,
        battery_kwh=60.0,
        efficiency_mi_per_kwh=3.5,
        plug_in_frequency_per_day=1.0,
        charger_kw=7.0,
        plug_in_hour=0.0,
        plug_out_hour=23.983,
        target_soc=0.80,
        plug_time_sigma_hours=0.0,
    ),
)

# Illustrative preset (not part of the surveyed population): a driver who does
# most of their mileage on weekend trips. Demonstrates the weekday/weekend
# mechanism the CNZ report motivates but the archetype table does not quantify.
WEEKEND_TRIPPER = Archetype(
    name="Weekend tripper (example)",
    population_share=0.0,
    annual_miles=9_435,
    battery_kwh=60.0,
    efficiency_mi_per_kwh=3.5,
    plug_in_frequency_per_day=1.0,
    charger_kw=7.0,
    plug_in_hour=18.0,
    plug_out_hour=7.0,
    target_soc=0.80,
    weekend_miles_multiplier=3.0,
)
