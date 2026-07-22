import json
from datetime import UTC, datetime

import httpx
import pytest

from axolotl.prices import (
    CHEAP_PRICE_P_PER_KWH,
    HALF_HOURS_PER_DAY,
    PEAK_PRICE_P_PER_KWH,
    _resample,
    fetch_agile_profile,
    get_price_profile,
    synthetic_price_profile,
)


def make_mock_client(prices_by_slot: list[float], days: int = 2) -> httpx.Client:
    """A client whose API responses cover `days` full days with given slot prices."""

    def handler(request: httpx.Request) -> httpx.Response:
        results = [
            {
                "valid_from": datetime(2026, 6, 1 + day, slot // 2, (slot % 2) * 30, tzinfo=UTC)
                .isoformat()
                .replace("+00:00", "Z"),
                "value_inc_vat": prices_by_slot[slot],
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


def test_fetch_agile_profile_averages_by_local_slot() -> None:
    prices = [float(slot) for slot in range(HALF_HOURS_PER_DAY)]
    profile = fetch_agile_profile(client=make_mock_client(prices))
    # June is BST (UTC+1): a rate valid from 00:00 UTC lands in the 01:00 local slot.
    assert profile[2] == 0.0
    assert profile[0] == 46.0


def test_fetch_agile_profile_rejects_incomplete_days() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        results = [{"valid_from": "2026-06-01T10:00:00Z", "value_inc_vat": 20.0}]
        return httpx.Response(200, text=json.dumps({"next": None, "results": results}))

    with pytest.raises(ValueError, match="incomplete"):
        fetch_agile_profile(client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_get_price_profile_falls_back_to_synthetic(monkeypatch: pytest.MonkeyPatch) -> None:
    def failing_fetch(*args: object, **kwargs: object) -> list[float]:
        raise httpx.ConnectError("offline")

    monkeypatch.setattr("axolotl.prices.fetch_agile_profile", failing_fetch)
    profile = get_price_profile(48)
    assert profile.source == "synthetic"
    assert profile.values_p_per_kwh == synthetic_price_profile(48)


def test_get_price_profile_without_live_uses_synthetic() -> None:
    profile = get_price_profile(24, use_live=False)
    assert profile.source == "synthetic"
    assert len(profile.values_p_per_kwh) == 24


def test_resample_down_averages_and_up_repeats() -> None:
    half_hourly = [float(slot) for slot in range(48)]
    assert _resample(half_hourly, 48) == half_hourly
    assert _resample(half_hourly, 24) == [i * 2 + 0.5 for i in range(24)]
    assert _resample(half_hourly, 96)[:4] == [0.0, 0.0, 1.0, 1.0]
    with pytest.raises(ValueError, match="resample"):
        _resample(half_hourly, 30)
