import pytest
from plotly.graph_objects import Figure

from axolotl.aggregate import time_of_day_profile
from axolotl.chart import build_agent_chart, build_population_chart
from axolotl.config import SimulationConfig
from axolotl.engine import SimulationResult, run_simulation
from axolotl.prices import synthetic_price_profile


@pytest.fixture(scope="module")
def result() -> SimulationResult:
    return run_simulation(SimulationConfig(n_agents=50, n_days=7))


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


def test_agent_chart_shows_trajectory_and_plug_events(result: SimulationResult) -> None:
    config = result.config
    fig = build_agent_chart(result, agent_index=0, n_days=5)
    names = [trace.name for trace in fig.data]
    assert names == ["Plugged in", "State of charge", "Plug-in event"]

    spd = config.steps_per_day
    first = config.burn_in_days * spd
    last = first + 5 * spd
    expected_events = (
        (result.plug_event_agent == 0)
        & (result.plug_event_step >= first)
        & (result.plug_event_step < last)
    ).sum()
    assert len(fig.data[2].x) == expected_events
    assert len(fig.data[1].y) == 5 * spd


def test_agent_chart_defaults_to_full_run_after_burn_in(result: SimulationResult) -> None:
    n_kept_days = result.config.n_days - result.config.burn_in_days
    expected_steps = n_kept_days * result.config.steps_per_day
    assert len(build_agent_chart(result, agent_index=3).data[1].y) == expected_steps
    assert len(build_agent_chart(result, agent_index=3, n_days=999).data[1].y) == expected_steps


def test_agent_chart_opens_on_one_day_with_range_picker(result: SimulationResult) -> None:
    fig = build_agent_chart(result, agent_index=0)
    x0, x1 = fig.layout.xaxis.range
    assert (x1 - x0).days == 1
    assert fig.layout.xaxis.rangeslider.visible
    # The 7-day fixture keeps 5 days after burn-in, so 1w/2w would duplicate All.
    assert [button.label for button in fig.layout.updatemenus[0].buttons] == ["1d", "3d", "All"]


def test_agent_chart_range_buttons_stay_within_the_data(result: SimulationResult) -> None:
    fig = build_agent_chart(result, agent_index=0)
    data_start = fig.data[1].x[0]
    step = fig.data[1].x[1] - data_start
    data_end = fig.data[1].x[-1] + step
    for button in fig.layout.updatemenus[0].buttons:
        range_start, range_end = button.args[0]["xaxis.range"]
        assert range_start == data_start
        assert range_end <= data_end


def test_agent_chart_with_prices_adds_price_panel(result: SimulationResult) -> None:
    spd = result.config.steps_per_day
    fig = build_agent_chart(
        result,
        agent_index=0,
        price_values=synthetic_price_profile(spd),
        price_source="synthetic",
    )
    price_traces = [t for t in fig.data if t.name and "price" in t.name.lower()]
    assert len(price_traces) == 1
    assert price_traces[0].yaxis == "y2"
    # Tiled across every displayed step, plus the closing point at the window end.
    n_kept_steps = (result.config.n_days - result.config.burn_in_days) * spd
    assert len(price_traces[0].y) == n_kept_steps + 1


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
