"""Interactive dashboard for the EV driver behaviour simulator."""

import streamlit as st

from axolotl.aggregate import DayFilter, time_of_day_profile
from axolotl.archetypes import ARCHETYPES, WEEKEND_TRIPPER, Archetype
from axolotl.chart import build_agent_chart, build_population_chart
from axolotl.config import SimulationConfig
from axolotl.engine import SimulationResult, run_simulation
from axolotl.prices import get_price_series, price_time_of_day_stats

ALL_ARCHETYPES: dict[str, Archetype] = {a.name: a for a in (*ARCHETYPES, WEEKEND_TRIPPER)}
DAY_FILTERS: dict[str, DayFilter] = {
    "All days": "all",
    "Weekdays": "weekday",
    "Weekends": "weekend",
}


def fmt_hour(hour: float) -> str:
    minutes = round(hour * 60)
    return f"{minutes // 60 % 24:02d}:{minutes % 60:02d}"


@st.cache_data(ttl=1800, show_spinner=False)
def cached_price_series(
    steps_per_day: int, n_days: int, use_live: bool
) -> tuple[list[float], str, str | None]:
    series = get_price_series(steps_per_day, n_days, use_live=use_live)
    start = series.start_date.isoformat() if series.start_date else None
    return series.values_p_per_kwh, series.source, start


# The full result (not just aggregates) is cached so the individual-driver
# view can read any agent's trajectory without re-simulating; max_entries
# bounds memory since each result holds the per-agent timestep arrays.
@st.cache_data(max_entries=8, show_spinner="Simulating the population…")
def cached_simulation(
    names: tuple[str, ...],
    n_agents: int,
    n_days: int,
    seed: int,
    spread: float,
    prices: tuple[float, ...],
) -> SimulationResult:
    config = SimulationConfig(n_agents=n_agents, n_days=n_days, seed=seed, spread=spread)
    return run_simulation(
        config,
        archetypes=[ALL_ARCHETYPES[name] for name in names],
        price_profile=list(prices),
    )


st.set_page_config(page_title="EV Driver Behaviour Simulator", page_icon="🔌", layout="wide")

with st.sidebar:
    st.header("Simulation inputs")
    st.subheader("Drivers")
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

    st.subheader("Prices")
    use_live_prices = st.toggle(
        "Live Octopus Agile prices",
        value=True,
        help="Real half-hourly rates, day by day over a recent Monday-aligned "
        "window matching the simulated weeks, used by smart charging. Falls "
        "back to a synthetic profile when the API is unreachable.",
    )

st.title("EV driver behaviour")
st.caption(
    "An agent-based population of EV drivers: when they are plugged in, and the "
    "state of charge of their batteries, over a typical day."
)

if not selected_names:
    st.warning("Select at least one archetype to simulate.")
    st.stop()

steps_per_day = SimulationConfig().steps_per_day
price_values, price_source, price_start = cached_price_series(
    steps_per_day, weeks * 7, use_live_prices
)
price_label = {"agile": "Octopus Agile", "synthetic": "synthetic"}[price_source]
result = cached_simulation(
    tuple(selected_names), n_agents, weeks * 7, seed, spread, tuple(price_values)
)

with st.container(border=True):
    st.subheader("All drivers")
    st.caption("When the fleet is plugged in, and its state of charge, pooled into a typical day.")
    with st.container(horizontal=True, vertical_alignment="bottom"):
        day_filter_label = (
            st.segmented_control("Days", list(DAY_FILTERS), default="All days") or "All days"
        )
        show_prices = st.toggle("Show price panel", value=True)

    profile = time_of_day_profile(result, DAY_FILTERS[day_filter_label])
    price_mean, price_p05, price_p95 = price_time_of_day_stats(price_values, steps_per_day)
    st.plotly_chart(
        build_population_chart(
            profile,
            price_values=price_mean if show_prices else None,
            price_source=price_label,
            price_band=(price_p05, price_p95) if show_prices else None,
        ),
        width="stretch",
    )

    if price_source == "agile":
        price_note = f"Octopus Agile half-hourly rates, day by day from Monday {price_start}"
    elif use_live_prices:
        price_note = "synthetic profile (Agile API unreachable)"
    else:
        price_note = "synthetic profile"
    st.caption(
        f"{n_agents:,} agents · {weeks} weeks at 30-minute resolution · "
        f"prices: {price_note} · smart chargers schedule into the cheapest slots."
    )

with st.container(border=True):
    st.subheader("Individual driver")
    st.caption("Pick one agent to inspect the behaviour the population view aggregates.")

    present_archetypes = sorted(
        {agent.archetype.name for agent in result.agents},
        key=[a.name for a in ALL_ARCHETYPES.values()].index,
    )
    with st.container(horizontal=True, vertical_alignment="bottom"):
        chosen_archetype = st.selectbox("Archetype", present_archetypes)
        driver_indices = [
            i for i, agent in enumerate(result.agents) if agent.archetype.name == chosen_archetype
        ]
        driver_number = int(
            st.number_input("Driver", min_value=1, max_value=len(driver_indices), value=1)
        )
        show_agent_prices = st.toggle("Show price panel", value=True, key="agent_prices")

    agent_index = driver_indices[driver_number - 1]
    agent = result.agents[agent_index]
    # The stable key lets the chart update in place, so uirevision can keep
    # the viewer's zoom when switching driver.
    st.plotly_chart(
        build_agent_chart(
            result,
            agent_index,
            price_values=price_values if show_agent_prices else None,
            price_source=price_label,
        ),
        width="stretch",
        key="agent_chart",
    )

    cadence = max(1, round(1 / agent.archetype.plug_in_frequency_per_day))
    weekend_times = (
        f"{fmt_hour(agent.weekend_plug_in_hour)} / {fmt_hour(agent.weekend_plug_out_hour)}"
    )
    st.caption(
        f"Driver {driver_number} of {len(driver_indices)} ({chosen_archetype}): "
        f"arrives home ~{fmt_hour(agent.plug_in_hour)}, leaves ~{fmt_hour(agent.plug_out_hour)} "
        f"(weekends {weekend_times}) · "
        f"~{agent.mean_daily_miles:.0f} miles/day · charges every "
        f"{'day' if cadence == 1 else f'{cadence} days'} to a {agent.target_soc:.0%} target."
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
        returns at its plug-in time. At weekends drivers arrive home about an
        hour earlier and leave about two hours later, with more spread — as
        observed in the study. While plugged in, most archetypes charge
        immediately at full power until their target state of charge; the
        *Intelligent Octopus* archetype instead schedules charging into the
        cheapest half-hours that still reach the target by its ready-by time.

        The chart pools all agents and days (after a burn-in period) into a
        typical-day view: bars show the share of the fleet plugged in, the line
        and band show the distribution of battery state of charge.
        """
    )
