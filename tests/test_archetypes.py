import pytest
from pydantic import ValidationError

from axolotl.archetypes import ARCHETYPES, WEEKEND_TRIPPER, Archetype, ChargingStrategy


def by_name(name: str) -> Archetype:
    return next(a for a in ARCHETYPES if a.name == name)


def test_population_shares_sum_to_one() -> None:
    assert sum(a.population_share for a in ARCHETYPES) == pytest.approx(1.0)


# Expected values are the archetype table's derived columns. Where the table
# rounds (kWh/plug-in) or blends in report medians rather than arithmetic
# (Intelligent Octopus plug-in SoC and charging duration), tolerances are
# loosened to the table's rounding precision.
@pytest.mark.parametrize(
    ("name", "kwh_per_year", "kwh_per_plug_in", "plug_in_soc", "charging_hours"),
    [
        ("Average (UK)", 2696, 7.0, 0.68, 1.0),
        ("Intelligent Octopus average", 8030, 22.0, 0.52, 2.5),
        ("Infrequent charging", 2696, 37.0, 0.18, 5.0),
        ("Infrequent driving", 1629, 4.0, 0.73, 1.0),
        ("Scheduled charging", 2696, 7.0, 0.68, 1.0),
        ("Always plugged-in", 2696, 7.0, 0.68, 1.0),
    ],
)
def test_derived_columns_match_archetype_table(
    name: str,
    kwh_per_year: float,
    kwh_per_plug_in: float,
    plug_in_soc: float,
    charging_hours: float,
) -> None:
    archetype = by_name(name)
    assert archetype.kwh_per_year == pytest.approx(kwh_per_year, rel=0.01)
    assert archetype.kwh_per_plug_in == pytest.approx(kwh_per_plug_in, abs=0.5)
    assert archetype.expected_plug_in_soc == pytest.approx(plug_in_soc, abs=0.03)
    assert archetype.expected_charging_hours == pytest.approx(charging_hours, abs=0.7)


def test_only_intelligent_octopus_charges_smart() -> None:
    smart = [a.name for a in ARCHETYPES if a.strategy is ChargingStrategy.SMART]
    assert smart == ["Intelligent Octopus average"]


def test_weekend_multiplier_preserves_weekly_miles() -> None:
    for archetype in (*ARCHETYPES, WEEKEND_TRIPPER):
        weekly = 5 * archetype.daily_miles_multiplier(is_weekend=False) + (
            2 * archetype.daily_miles_multiplier(is_weekend=True)
        )
        assert weekly == pytest.approx(7.0)


def test_invalid_archetype_values_rejected() -> None:
    base = by_name("Average (UK)")
    with pytest.raises(ValidationError, match="population_share"):
        Archetype.model_validate({**base.model_dump(), "population_share": 1.5})
    with pytest.raises(ValidationError, match="target_soc"):
        Archetype.model_validate({**base.model_dump(), "target_soc": 0.0})
    with pytest.raises(ValidationError, match="plug_in_frequency_per_day"):
        Archetype.model_validate({**base.model_dump(), "plug_in_frequency_per_day": 0.0})
    with pytest.raises(ValidationError, match="weekend_miles_multiplier"):
        Archetype.model_validate({**base.model_dump(), "weekend_miles_multiplier": 4.0})
