"""Driver archetypes: population-level presets the simulation samples agents from.

The six presets transcribe the archetype table provided with the task, itself
derived from Centre for Net Zero's "Learning from Intelligent Octopus" report
(May 2022). Times are local hours; SoC values are fractions of battery capacity.
"""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

DAYS_PER_YEAR = 365
WEEKDAYS_PER_WEEK = 5
WEEKEND_DAYS_PER_WEEK = 2
# Above this, weekday miles would have to be negative to preserve annual mileage.
MAX_WEEKEND_MILES_MULTIPLIER = 7 / WEEKEND_DAYS_PER_WEEK

# Per-agent charging-target preferences, as (target SoC, probability). CNZ
# report Fig. 2: the three most common preferences are 80%, 90% and 100%
# (25/24/23% of users); the remaining 28% mostly sit below 80% and are
# bucketed here at 70%. Population mean ~0.84 — close to the archetype
# table's flat 0.8, so its derived plug-in SoC figures remain recapitulated.
TARGET_SOC_PREFERENCES: tuple[tuple[float, float], ...] = (
    (0.7, 0.28),
    (0.8, 0.25),
    (0.9, 0.24),
    (1.0, 0.23),
)


class ChargingStrategy(StrEnum):
    """How the car charges while plugged in."""

    # Charge at full charger power from plug-in until target SoC is reached.
    IMMEDIATE = "immediate"
    # Charge in the cheapest half-hour slots so target SoC is reached by
    # `ready_by_hour` (Intelligent Octopus-style automation).
    SMART = "smart"


class Archetype(BaseModel):
    """Parameters describing one behavioural pattern of EV drivers.

    Per-agent variation is applied on top of these means when sampling a
    population (see `agent.sample_population`).
    """

    model_config = ConfigDict(frozen=True)

    name: str
    population_share: float = Field(ge=0, le=1)
    annual_miles: float = Field(gt=0)
    battery_kwh: float = Field(gt=0)
    efficiency_mi_per_kwh: float = Field(gt=0)
    # How often the driver plugs in (1.0 = daily, 0.2 = every 5th day).
    plug_in_frequency_per_day: float = Field(gt=0, le=1)
    charger_kw: float = Field(gt=0)
    # Mean plug-in/plug-out times, fractional local hours (18.5 = 18:30).
    plug_in_hour: float = Field(ge=0, lt=24)
    plug_out_hour: float = Field(ge=0, lt=24)
    target_soc: float = Field(gt=0, le=1)
    strategy: ChargingStrategy = ChargingStrategy.IMMEDIATE
    # SMART only: deadline by which target SoC should be reached.
    ready_by_hour: float = Field(default=7.0, ge=0, lt=24)
    # Std dev of per-agent plug-in/out times. ~1h spread around the evening
    # peak per CNZ report Fig. 4; 0 disables time variation.
    plug_time_sigma_hours: float = Field(default=1.0, ge=0)
    # Weekend timing shifts relative to the agent's weekday habit: plug in
    # ~1h earlier, plug out ~2h later (CNZ report Figs. 4-5).
    weekend_plug_in_shift_hours: float = Field(default=-1.0, ge=-12, le=12)
    weekend_plug_out_shift_hours: float = Field(default=2.0, ge=-12, le=12)
    # Lognormal sigma of per-agent mean daily mileage around the archetype mean.
    miles_sigma: float = Field(default=0.25, ge=0)
    # Weekend daily miles relative to the archetype's average day. Weekday
    # miles are scaled so annual mileage is preserved (see multiplier methods).
    weekend_miles_multiplier: float = Field(default=1.0, ge=0, le=MAX_WEEKEND_MILES_MULTIPLIER)

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
        weekend_plug_in_shift_hours=0.0,
        weekend_plug_out_shift_hours=0.0,
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
