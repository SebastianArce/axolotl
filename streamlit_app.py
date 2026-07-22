"""Interactive dashboard for the EV driver behaviour simulator."""

import polars as pl
import streamlit as st

from axolotl.aggregate import DayFilter, time_of_day_profile
from axolotl.archetypes import ARCHETYPES, WEEKEND_TRIPPER, Archetype
from axolotl.chart import build_population_chart
from axolotl.config import SimulationConfig
from axolotl.engine import run_simulation
from axolotl.prices import get_price_profile

ALL_ARCHETYPES: dict[str, Archetype] = {a.name: a for a in (*ARCHETYPES, WEEKEND_TRIPPER)}
DAY_FILTERS: dict[str, DayFilter] = {
    "All days": "all",
    "Weekdays": "weekday",
    "Weekends": "weekend",
}


@st.cache_data(ttl=1800, show_spinner=False)
def cached_prices(steps_per_day: int, use_live: bool) -> tuple[list[float], str]:
    profile = get_price_profile(steps_per_day, use_live=use_live)
    return profile.values_p_per_kwh, profile.source


@st.cache_data(max_entries=32, show_spinner="Simulating the population…")
def cached_profiles(
    names: tuple[str, ...],
    n_agents: int,
    n_days: int,
    seed: int,
    spread: float,
    prices: tuple[float, ...],
) -> dict[str, pl.DataFrame]:
    config = SimulationConfig(n_agents=n_agents, n_days=n_days, seed=seed, spread=spread)
    result = run_simulation(
        config,
        archetypes=[ALL_ARCHETYPES[name] for name in names],
        price_profile=list(prices),
    )
    return {
        day_filter: time_of_day_profile(result, day_filter)
        for day_filter in ("all", "weekday", "weekend")
    }


st.set_page_config(page_title="EV Driver Behaviour Simulator", page_icon="🔌", layout="wide")

with st.sidebar:
    st.header("Population")
    selected_names = st.multiselect(
        "Archetypes",
        options=list(ALL_ARCHETYPES),
        default=[a.name for a in ARCHETYPES],
        help="Population shares are renormalised over the selection.",
    )
    n_agents = st.slider("Agents", min_value=100, max_value=3000, value=1000, step=100)
    weeks = st.slider("Simulated weeks", min_value=1, max_value=8, value=4)
    spread = st.slider(
        "Behavioural spread",
        min_value=0.0,
        max_value=2.0,
        value=1.0,
        step=0.1,
        help="Scales how much individual drivers vary around their archetype "
        "(plug-in times, mileage). 0 = every driver behaves exactly like its archetype.",
    )
    seed = int(st.number_input("Random seed", min_value=0, value=42, step=1))

    st.header("View")
    day_filter_label = (
        st.segmented_control("Days", list(DAY_FILTERS), default="All days") or "All days"
    )
    use_live_prices = st.toggle(
        "Live Octopus Agile prices",
        value=True,
        help="28-day average by time of day. Falls back to a synthetic "
        "profile when the API is unreachable.",
    )
    show_prices = st.toggle("Show price panel", value=True)

st.title("EV driver behaviour")
st.caption(
    "An agent-based population of EV drivers: when they are plugged in, and the "
    "state of charge of their batteries, over a typical day."
)

if not selected_names:
    st.warning("Select at least one archetype to simulate.")
    st.stop()

steps_per_day = SimulationConfig().steps_per_day
price_values, price_source = cached_prices(steps_per_day, use_live_prices)
profiles = cached_profiles(
    tuple(selected_names), n_agents, weeks * 7, seed, spread, tuple(price_values)
)
profile = profiles[DAY_FILTERS[day_filter_label]]

st.plotly_chart(
    build_population_chart(
        profile,
        price_values=price_values if show_prices else None,
        price_source={"agile": "Octopus Agile", "synthetic": "synthetic"}[price_source],
    ),
    width="stretch",
)

price_note = (
    "Octopus Agile, 28-day average by time of day"
    if price_source == "agile"
    else "synthetic profile (Agile API unreachable)"
)
st.caption(
    f"{n_agents:,} agents · {weeks} weeks at 30-minute resolution · "
    f"prices: {price_note} · smart chargers schedule into the cheapest slots."
)

with st.expander("How the simulation works"):
    st.markdown(
        """
        Each **agent** is one driver, sampled from an archetype with individual
        variation (Monte Carlo): habitual plug-in/out times, mean daily mileage,
        a charging cadence (daily, or every *k* days), and a personal charging
        target (70/80/90/100%, following the preference mix observed in the
        Centre for Net Zero's Intelligent Octopus study).

        Each simulated day the car leaves at its plug-out time, drives a
        gamma-distributed number of miles (weekday/weekend adjusted), and
        returns at its plug-in time. While plugged in, most archetypes charge
        immediately at full power until their target state of charge; the
        *Intelligent Octopus* archetype instead schedules charging into the
        cheapest half-hours that still reach the target by its ready-by time.

        The chart pools all agents and days (after a burn-in period) into a
        typical-day view: bars show the share of the fleet plugged in, the line
        and band show the distribution of battery state of charge.
        """
    )
