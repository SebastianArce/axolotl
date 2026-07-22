"""Electricity price profiles used by the SMART charging strategy.

For now this provides a synthetic time-of-use profile shaped like a typical GB
day: an evening peak, moderate daytime prices, and a cheap overnight window
matching the Intelligent Octopus window (23:30-05:30, per the CNZ report).
"""

CHEAP_WINDOW_START_HOUR = 23.5
CHEAP_WINDOW_END_HOUR = 5.5
PEAK_START_HOUR = 16.0
PEAK_END_HOUR = 19.0

CHEAP_PRICE_P_PER_KWH = 7.5  # Intelligent Octopus overnight rate
DAY_PRICE_P_PER_KWH = 25.0
PEAK_PRICE_P_PER_KWH = 35.0


def default_price_profile(steps_per_day: int) -> list[float]:
    """Synthetic p/kWh price for each timestep of a day."""
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
