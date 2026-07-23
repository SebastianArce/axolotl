"""Electricity prices used by the SMART charging strategy.

Two sources:

- Octopus Agile: real half-hourly prices from the public Octopus Energy API,
  taken day by day over a recent Monday-aligned window so simulated weekdays
  and weekends line up with real ones. Agile follows day-ahead wholesale
  prices, so this is the natural signal for Intelligent Octopus-style smart
  charging — and its day-to-day volatility is what makes the scheduling
  visibly adaptive.
- Synthetic fallback: a deterministic GB-shaped profile (evening peak, cheap
  23:30-05:30 overnight window per the CNZ report), identical every day, used
  when the API is unavailable and by tests that need reproducibility without
  network access.
"""

from datetime import UTC, date, datetime, timedelta
from typing import Literal
from zoneinfo import ZoneInfo

import httpx
import numpy as np
from pydantic import BaseModel, ConfigDict

AGILE_PRODUCT = "AGILE-24-10-01"
# Region C = London. Regional Agile prices differ by a roughly constant offset,
# so the *shape* — which is all smart scheduling cares about — is unaffected.
AGILE_REGION = "C"
LOCAL_TZ = ZoneInfo("Europe/London")
HALF_HOURS_PER_DAY = 48

CHEAP_WINDOW_START_HOUR = 23.5
CHEAP_WINDOW_END_HOUR = 5.5
PEAK_START_HOUR = 16.0
PEAK_END_HOUR = 19.0

CHEAP_PRICE_P_PER_KWH = 7.5  # Intelligent Octopus overnight rate
DAY_PRICE_P_PER_KWH = 25.0
PEAK_PRICE_P_PER_KWH = 35.0


class PriceSeries(BaseModel):
    """A sequential per-timestep price series covering whole days.

    `start_date` is the real Monday the window begins on for Agile data, and
    None for the synthetic profile (the simulation itself is dateless).
    """

    model_config = ConfigDict(frozen=True)

    values_p_per_kwh: list[float]
    source: Literal["agile", "synthetic"]
    steps_per_day: int
    start_date: date | None = None


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


def synthetic_price_series(steps_per_day: int, n_days: int) -> list[float]:
    """The synthetic day repeated: deliberately flat day-to-day, so offline
    runs and calibration tests stay deterministic."""
    return synthetic_price_profile(steps_per_day) * n_days


def get_price_series(steps_per_day: int, n_days: int, use_live: bool = True) -> PriceSeries:
    """Day-by-day Agile series when the API is reachable, synthetic otherwise."""
    if use_live:
        today = datetime.now(tz=LOCAL_TZ).date()
        start = monday_aligned_start(n_days, today)
        try:
            half_hourly = fetch_agile_series(start, n_days)
            return PriceSeries(
                values_p_per_kwh=_resample_series(half_hourly, steps_per_day),
                source="agile",
                steps_per_day=steps_per_day,
                start_date=start,
            )
        except (httpx.HTTPError, ValueError):
            pass
    return PriceSeries(
        values_p_per_kwh=synthetic_price_series(steps_per_day, n_days),
        source="synthetic",
        steps_per_day=steps_per_day,
    )


def monday_aligned_start(n_days: int, today: date) -> date:
    """Start of the most recent Monday-aligned `n_days` window fully in the past.

    Rolling the anchor back to its Monday keeps the window's end on or before
    today (only fully published days are used) and lines simulated day 0 — a
    Monday — up with a real Monday, so weekend price shapes match.
    """
    anchor = today - timedelta(days=n_days)
    return anchor - timedelta(days=anchor.weekday())


def fetch_agile_series(
    start: date,
    n_days: int,
    client: httpx.Client | None = None,
) -> list[float]:
    """Sequential half-hourly Agile rates for `n_days` starting at `start`.

    Rates come back in UTC and are bucketed by (local date, half-hour slot) in
    Europe/London. The simulation's day is an idealized 48-slot local day, so
    DST days are normalized: the autumn clock change's doubled 01:xx slots
    average into one bucket, and the spring change's missing slots are filled
    from the same slot on the nearest day that has one.
    """
    # ±1 day of UTC margin so local-time bucketing sees the window's edges.
    period_from = datetime.combine(start, datetime.min.time(), tzinfo=UTC) - timedelta(days=1)
    period_to = period_from + timedelta(days=n_days + 2)

    sums = [[0.0] * HALF_HOURS_PER_DAY for _ in range(n_days)]
    counts = [[0] * HALF_HOURS_PER_DAY for _ in range(n_days)]
    for valid_from, price in _fetch_agile_rates(period_from, period_to, client):
        local = valid_from.astimezone(LOCAL_TZ)
        day = (local.date() - start).days
        if 0 <= day < n_days:
            slot = local.hour * 2 + local.minute // 30
            sums[day][slot] += price
            counts[day][slot] += 1

    series = [
        [total / count if count else None for total, count in zip(s, c, strict=True)]
        for s, c in zip(sums, counts, strict=True)
    ]
    return _fill_missing_slots(series)


def _fill_missing_slots(series: list[list[float | None]]) -> list[float]:
    """Fill each empty slot from the same slot on the nearest day that has one."""
    n_days = len(series)
    filled: list[float] = []
    for day, day_prices in enumerate(series):
        for slot, price in enumerate(day_prices):
            if price is None:
                for offset in range(1, n_days):
                    for other in (day - offset, day + offset):
                        if 0 <= other < n_days and series[other][slot] is not None:
                            price = series[other][slot]
                            break
                    if price is not None:
                        break
            if price is None:
                raise ValueError("Agile API returned incomplete time-of-day coverage")
            filled.append(price)
    return filled


def price_time_of_day_stats(
    values: list[float], steps_per_day: int
) -> tuple[list[float], list[float], list[float]]:
    """Per-slot (mean, 5th, 95th percentile) across the days of a series."""
    by_day = np.asarray(values).reshape(-1, steps_per_day)
    return (
        by_day.mean(axis=0).tolist(),
        np.percentile(by_day, 5, axis=0).tolist(),
        np.percentile(by_day, 95, axis=0).tolist(),
    )


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


def _resample_series(half_hourly: list[float], steps_per_day: int) -> list[float]:
    """Resample each 48-slot day block of a sequential series."""
    resampled: list[float] = []
    for day_start in range(0, len(half_hourly), HALF_HOURS_PER_DAY):
        resampled.extend(
            _resample(half_hourly[day_start : day_start + HALF_HOURS_PER_DAY], steps_per_day)
        )
    return resampled


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
