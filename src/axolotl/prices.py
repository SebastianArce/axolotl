"""Electricity price profiles used by the SMART charging strategy.

Two sources:

- Octopus Agile: real half-hourly prices from the public Octopus Energy API,
  averaged over a recent window into a typical time-of-day profile. Agile
  follows day-ahead wholesale prices, so this is the natural signal for
  Intelligent Octopus-style smart charging.
- Synthetic fallback: a deterministic GB-shaped profile (evening peak, cheap
  23:30-05:30 overnight window per the CNZ report) used when the API is
  unavailable, and by tests that need reproducibility without network access.
"""

from datetime import UTC, datetime, timedelta
from typing import Literal
from zoneinfo import ZoneInfo

import httpx
from pydantic import BaseModel, ConfigDict

AGILE_PRODUCT = "AGILE-24-10-01"
# Region C = London. Regional Agile prices differ by a roughly constant offset,
# so the *shape* — which is all smart scheduling cares about — is unaffected.
AGILE_REGION = "C"
AGILE_LOOKBACK_DAYS = 28
LOCAL_TZ = ZoneInfo("Europe/London")
HALF_HOURS_PER_DAY = 48

CHEAP_WINDOW_START_HOUR = 23.5
CHEAP_WINDOW_END_HOUR = 5.5
PEAK_START_HOUR = 16.0
PEAK_END_HOUR = 19.0

CHEAP_PRICE_P_PER_KWH = 7.5  # Intelligent Octopus overnight rate
DAY_PRICE_P_PER_KWH = 25.0
PEAK_PRICE_P_PER_KWH = 35.0


class PriceProfile(BaseModel):
    """A per-timestep-of-day price profile and where it came from."""

    model_config = ConfigDict(frozen=True)

    values_p_per_kwh: list[float]
    source: Literal["agile", "synthetic"]


def synthetic_price_profile(steps_per_day: int) -> list[float]:
    """Deterministic GB-shaped p/kWh price for each timestep of a day."""
    prices = []
    for step in range(steps_per_day):
        hour = step * 24 / steps_per_day
        if hour >= CHEAP_WINDOW_START_HOUR or hour < CHEAP_WINDOW_END_HOUR:
            prices.append(CHEAP_PRICE_P_PER_KWH)
        elif PEAK_START_HOUR <= hour < PEAK_END_HOUR:
            prices.append(PEAK_PRICE_P_PER_KWH)
        else:
            prices.append(DAY_PRICE_P_PER_KWH)
    return prices


def get_price_profile(steps_per_day: int, use_live: bool = True) -> PriceProfile:
    """Agile-based profile when the API is reachable, synthetic otherwise."""
    if use_live:
        try:
            return PriceProfile(
                values_p_per_kwh=_resample(fetch_agile_profile(), steps_per_day),
                source="agile",
            )
        except (httpx.HTTPError, ValueError):
            pass
    return PriceProfile(values_p_per_kwh=synthetic_price_profile(steps_per_day), source="synthetic")


def fetch_agile_profile(
    client: httpx.Client | None = None,
    lookback_days: int = AGILE_LOOKBACK_DAYS,
) -> list[float]:
    """Average recent Agile half-hourly rates into a 48-slot time-of-day profile.

    Rates come back in UTC; slots are bucketed in Europe/London local time so
    the overnight-cheap shape lines up with the simulation's local-time day.
    """
    period_to = datetime.now(tz=UTC).replace(minute=0, second=0, microsecond=0)
    period_from = period_to - timedelta(days=lookback_days)

    sums = [0.0] * HALF_HOURS_PER_DAY
    counts = [0] * HALF_HOURS_PER_DAY
    for valid_from, price in _fetch_agile_rates(period_from, period_to, client):
        local = valid_from.astimezone(LOCAL_TZ)
        slot = local.hour * 2 + local.minute // 30
        sums[slot] += price
        counts[slot] += 1

    if any(count == 0 for count in counts):
        raise ValueError("Agile API returned incomplete time-of-day coverage")
    return [total / count for total, count in zip(sums, counts, strict=True)]


def _fetch_agile_rates(
    period_from: datetime,
    period_to: datetime,
    client: httpx.Client | None = None,
) -> list[tuple[datetime, float]]:
    """Fetch half-hourly Agile unit rates (p/kWh inc VAT), following pagination."""
    url = (
        f"https://api.octopus.energy/v1/products/{AGILE_PRODUCT}/electricity-tariffs/"
        f"E-1R-{AGILE_PRODUCT}-{AGILE_REGION}/standard-unit-rates/"
    )
    params: dict[str, str | int] | None = {
        "period_from": period_from.isoformat(),
        "period_to": period_to.isoformat(),
        "page_size": 1500,
    }

    owns_client = client is None
    if client is None:
        # Transport retries cover transient connection failures (DNS, dropped
        # handshakes) without delaying the synthetic fallback when the API is
        # genuinely down: HTTP error responses are not retried.
        client = httpx.Client(timeout=10, transport=httpx.HTTPTransport(retries=2))
    rates: list[tuple[datetime, float]] = []
    try:
        while url:
            response = client.get(url, params=params)
            response.raise_for_status()
            payload = response.json()
            rates.extend(
                (datetime.fromisoformat(r["valid_from"]), float(r["value_inc_vat"]))
                for r in payload["results"]
            )
            url = payload["next"]
            params = None  # the `next` URL already carries the query string
    finally:
        if owns_client:
            client.close()
    return rates


def _resample(half_hourly: list[float], steps_per_day: int) -> list[float]:
    """Adapt a 48-slot profile to the simulation's timestep count."""
    if steps_per_day == HALF_HOURS_PER_DAY:
        return list(half_hourly)
    if steps_per_day < HALF_HOURS_PER_DAY and HALF_HOURS_PER_DAY % steps_per_day == 0:
        group = HALF_HOURS_PER_DAY // steps_per_day
        return [sum(half_hourly[i * group : (i + 1) * group]) / group for i in range(steps_per_day)]
    if steps_per_day % HALF_HOURS_PER_DAY == 0:
        repeat = steps_per_day // HALF_HOURS_PER_DAY
        return [price for price in half_hourly for _ in range(repeat)]
    raise ValueError(f"cannot resample 48 half-hours to {steps_per_day} steps per day")
