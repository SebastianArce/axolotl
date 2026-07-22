from plotly.graph_objects import Figure

from axolotl.aggregate import time_of_day_profile
from axolotl.chart import build_population_chart
from axolotl.config import SimulationConfig
from axolotl.engine import run_simulation
from axolotl.prices import synthetic_price_profile


def make_profile():
    config = SimulationConfig(n_agents=50, n_days=7)
    return time_of_day_profile(run_simulation(config)), config.steps_per_day


def test_chart_has_bar_band_and_mean_traces() -> None:
    profile, _ = make_profile()
    fig = build_population_chart(profile)
    assert isinstance(fig, Figure)
    names = [trace.name for trace in fig.data if trace.name]
    assert "Plugged in" in names
    assert "Mean state of charge" in names
    assert "SoC 25–75th pct" in names
    assert "SoC 5–95th pct" in names


def test_chart_without_prices_has_single_panel() -> None:
    profile, _ = make_profile()
    fig = build_population_chart(profile)
    assert all(trace.name is None or "price" not in trace.name.lower() for trace in fig.data)


def test_chart_with_prices_adds_price_panel() -> None:
    profile, steps_per_day = make_profile()
    fig = build_population_chart(
        profile,
        price_values=synthetic_price_profile(steps_per_day),
        price_source="synthetic",
    )
    price_traces = [t for t in fig.data if t.name and "price" in t.name.lower()]
    assert len(price_traces) == 1
    assert price_traces[0].yaxis == "y2"
