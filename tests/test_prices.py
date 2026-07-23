import json
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta

import httpx
import pytest

from axolotl.prices import (
    CHEAP_PRICE_P_PER_KWH,
    HALF_HOURS_PER_DAY,
    PEAK_PRICE_P_PER_KWH,
    _resample,
    _resample_series,
    fetch_agile_series,
    get_price_series,
    monday_aligned_start,
    price_time_of_day_stats,
    synthetic_price_profile,
    synthetic_price_series,
)

START = date(2026, 6, 1)  # a Monday, in BST (UTC+1)


def make_mock_client(
    price_for: Callable[[int, int], float], start: date = START, days: int = 7
) -> httpx.Client:
    """A client whose responses cover `days` full local days from `start`,
    with each half-hour's price given by `price_for(day, slot)`."""

    def handler(request: httpx.Request) -> httpx.Response:
        first_local = datetime(start.year, start.month, start.day, tzinfo=UTC) - timedelta(hours=1)
        results = [
            {
                "valid_from": (first_local + timedelta(days=day, minutes=30 * slot))
                .isoformat()
                .replace("+00:00", "Z"),
                "value_inc_vat": price_for(day, slot),
            }
            for day in range(days)
            for slot in range(HALF_HOURS_PER_DAY)
        ]
        return httpx.Response(200, text=json.dumps({"next": None, "results": results}))

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_synthetic_profile_shape() -> None:
    profile = synthetic_price_profile(48)
    assert len(profile) == 48
    assert profile[0] == CHEAP_PRICE_P_PER_KWH  # 00:00 is in the overnight window
    assert profile[34] == PEAK_PRICE_P_PER_KWH  # 17:00 is in the evening peak
    assert min(profile) == CHEAP_PRICE_P_PER_KWH


def test_synthetic_series_repeats_the_day() -> None:
    series = synthetic_price_series(48, 3)
    assert series == synthetic_price_profile(48) * 3


def test_monday_aligned_start_is_a_full_past_window() -> None:
    for today in (date(2026, 7, 20), date(2026, 7, 22), date(2026, 7, 26)):  # Mon, Wed, Sun
        for n_days in (7, 28, 56):
            start = monday_aligned_start(n_days, today)
            assert start.weekday() == 0
            assert start + timedelta(days=n_days) <= today


def test_fetch_agile_series_is_sequential_by_local_day_and_slot() -> None:
    # The mock feed starts at 23:00 UTC == 00:00 BST, so correct local
    # bucketing maps feed (day, slot) exactly onto (local day, slot) — while
    # UTC bucketing would shift everything by two slots and fail.
    series = fetch_agile_series(
        START, 7, client=make_mock_client(lambda day, slot: day * 100.0 + slot)
    )
    assert len(series) == 7 * HALF_HOURS_PER_DAY
    assert series[0] == 0.0  # local Mon 00:00
    assert series[HALF_HOURS_PER_DAY - 1] == 47.0  # local Mon 23:30
    assert series[HALF_HOURS_PER_DAY] == 100.0  # local Tue 00:00
    assert series[-1] == 647.0  # local Sun 23:30


def test_fetch_agile_series_fills_missing_slot_from_nearest_day() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        first_local = datetime(2026, 1, 5, tzinfo=UTC)  # GMT: local == UTC
        results = [
            {
                "valid_from": (first_local + timedelta(days=day, minutes=30 * slot))
                .isoformat()
                .replace("+00:00", "Z"),
                "value_inc_vat": float(day),
            }
            for day in range(3)
            for slot in range(HALF_HOURS_PER_DAY)
            if not (day == 1 and slot == 10)  # drop one half-hour on day 1
        ]
        return httpx.Response(200, text=json.dumps({"next": None, "results": results}))

    series = fetch_agile_series(
        date(2026, 1, 5), 3, client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    assert series[HALF_HOURS_PER_DAY + 10] == 0.0  # filled from day 0's slot 10


def test_fetch_agile_series_rejects_slot_missing_everywhere() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        results = [{"valid_from": "2026-01-05T10:00:00Z", "value_inc_vat": 20.0}]
        return httpx.Response(200, text=json.dumps({"next": None, "results": results}))

    with pytest.raises(ValueError, match="incomplete"):
        fetch_agile_series(
            date(2026, 1, 5), 2, client=httpx.Client(transport=httpx.MockTransport(handler))
        )


def test_get_price_series_falls_back_to_synthetic(monkeypatch: pytest.MonkeyPatch) -> None:
    def failing_fetch(*args: object, **kwargs: object) -> list[float]:
        raise httpx.ConnectError("offline")

    monkeypatch.setattr("axolotl.prices.fetch_agile_series", failing_fetch)
    series = get_price_series(48, 7)
    assert series.source == "synthetic"
    assert series.start_date is None
    assert series.values_p_per_kwh == synthetic_price_series(48, 7)


def test_get_price_series_without_live_uses_synthetic() -> None:
    series = get_price_series(24, 7, use_live=False)
    assert series.source == "synthetic"
    assert len(series.values_p_per_kwh) == 24 * 7


def test_price_time_of_day_stats() -> None:
    # Two 4-slot days: slot means are the midpoints, percentiles bracket them.
    mean, p05, p95 = price_time_of_day_stats([0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0], 4)
    assert mean == [5.0, 6.0, 7.0, 8.0]
    assert p05 == [0.5, 1.5, 2.5, 3.5]
    assert p95 == [9.5, 10.5, 11.5, 12.5]


def test_resample_down_averages_and_up_repeats() -> None:
    half_hourly = [float(slot) for slot in range(48)]
    assert _resample(half_hourly, 48) == half_hourly
    assert _resample(half_hourly, 24) == [i * 2 + 0.5 for i in range(24)]
    assert _resample(half_hourly, 96)[:4] == [0.0, 0.0, 1.0, 1.0]
    with pytest.raises(ValueError, match="resample"):
        _resample(half_hourly, 30)


def test_resample_series_treats_each_day_separately() -> None:
    two_days = [float(slot) for slot in range(48)] + [float(slot) * 10 for slot in range(48)]
    resampled = _resample_series(two_days, 24)
    assert len(resampled) == 48
    assert resampled[0] == 0.5
    assert resampled[24] == 5.0
