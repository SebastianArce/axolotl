"""The dashboard charts.

Population chart: % plugged in and state of charge over a typical day. Both
series are percentages, so they honestly share a single 0-100% axis — no
dual-axis tricks. Electricity prices are a different unit (p/kWh) and get
their own slim panel below, aligned on the same time axis.

Agent chart: one driver's actual state-of-charge trajectory over consecutive
simulated days, with plugged-in periods shaded and each plug-in event marked
at the SoC it happened — the individual-level output the population
aggregates are built from. The same price panel sits below with each
displayed day's actual prices, so both views read the same way.

Design notes: the bars are deliberately translucent so the state-of-charge
story reads first; direct labels are sparse by intent and everything else
lives in the unified hover and the legend.
"""

from datetime import datetime, timedelta

import polars as pl
from plotly.graph_objects import Bar, Figure, Scatter

from axolotl.engine import SimulationResult

# Categorical palette slots (fixed assignment: the entity keeps its hue).
PLUGGED_IN_COLOR = "rgba(42, 120, 214, 0.45)"  # blue, quiet
PLUGGED_IN_WASH = "rgba(42, 120, 214, 0.14)"  # agent chart: plugged-in periods
SOC_COLOR = "#eb6834"  # orange
SOC_BAND_OUTER = "rgba(235, 104, 52, 0.10)"  # 5-95th percentile wash
SOC_BAND_INNER = "rgba(235, 104, 52, 0.16)"  # 25-75th percentile wash
PRICE_COLOR = "#1baf7a"  # aqua
PRICE_FILL = "rgba(27, 175, 122, 0.08)"
PRICE_BAND = "rgba(27, 175, 122, 0.14)"  # day-to-day 5-95th percentile wash

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

FONT_FAMILY = 'system-ui, -apple-system, "Segoe UI", sans-serif'
ANNOTATION_FONT = {"family": FONT_FAMILY, "size": 12, "color": INK_SECONDARY}


def build_population_chart(
    profile: pl.DataFrame,
    price_values: list[float] | None = None,
    price_source: str | None = None,
    price_band: tuple[list[float], list[float]] | None = None,
) -> Figure:
    """Build the dashboard figure from a time-of-day profile.

    `profile` is the output of `aggregate.time_of_day_profile`. When
    `price_values` is given (one per timestep of the day — the slot-wise mean
    of the day-by-day series), a price panel is added below; `price_band`
    (lower, upper per slot) adds a day-to-day spread wash around it, the price
    panel's counterpart to the SoC percentile bands.
    """
    hours = profile["hour"].to_list()
    step = hours[1] - hours[0] if len(hours) > 1 else 0.5
    centers = [h + step / 2 for h in hours]

    with_prices = price_values is not None
    # Both panels live on ONE x-axis as stacked y-domains (rather than
    # make_subplots' separate axes) so `hoversubplots="axis"` can show the
    # unified hover across the SoC panel and the price panel together.
    fig = Figure()

    _add_plugged_in_bars(fig, centers, profile, step)
    _add_soc_layers(fig, centers, profile)
    if with_prices:
        _add_price_panel(fig, hours, price_values, price_source, price_band)
    _style(fig, with_prices)
    return fig


def _add_plugged_in_bars(
    fig: Figure, centers: list[float], profile: pl.DataFrame, step: float
) -> None:
    fig.add_trace(
        Bar(
            x=centers,
            y=profile["pct_plugged_in"],
            name="Plugged in",
            legendrank=1,
            marker_color=PLUGGED_IN_COLOR,
            width=step * 0.66,
            hovertemplate="%{y:.1f}% of fleet<extra>Plugged in</extra>",
        )
    )


def _add_soc_layers(fig: Figure, centers: list[float], profile: pl.DataFrame) -> None:
    """Two nested percentile washes and the mean line on top."""
    bands = [
        ("soc_p95", "soc_p05", SOC_BAND_OUTER, "SoC 5–95th pct", 4),
        ("soc_p75", "soc_p25", SOC_BAND_INNER, "SoC 25–75th pct", 3),
    ]
    for upper, lower, fill, name, rank in bands:
        fig.add_trace(
            Scatter(
                x=centers,
                y=profile[upper],
                mode="lines",
                line={"width": 0},
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            Scatter(
                x=centers,
                y=profile[lower],
                mode="lines",
                line={"width": 0},
                fill="tonexty",
                fillcolor=fill,
                name=name,
                legendrank=rank,
                hoverinfo="skip",
            )
        )

    fig.add_trace(
        Scatter(
            x=centers,
            y=profile["soc_mean"],
            mode="lines",
            line={"width": 2.5, "color": SOC_COLOR, "shape": "spline"},
            name="Mean state of charge",
            legendrank=2,
            customdata=list(
                zip(
                    profile["soc_p05"],
                    profile["soc_p25"],
                    profile["soc_p75"],
                    profile["soc_p95"],
                    strict=True,
                )
            ),
            hovertemplate=(
                "%{y:.1f}% mean · 25–75th: %{customdata[1]:.0f}–%{customdata[2]:.0f}% · "
                "5–95th: %{customdata[0]:.0f}–%{customdata[3]:.0f}%"
                "<extra>State of charge</extra>"
            ),
        )
    )


def _add_price_panel(
    fig: Figure,
    hours: list[float],
    price_values: list[float],
    price_source: str | None,
    price_band: tuple[list[float], list[float]] | None = None,
) -> None:
    # Repeat the last value at 24:00 so the step line spans the full day and
    # both panels share an identical x extent.
    def closed(values: list[float]) -> list[float]:
        return [*values, values[-1]]

    lower, upper = price_band if price_band is not None else (None, None)
    if lower is not None and upper is not None:
        fig.add_trace(
            Scatter(
                x=[*hours, 24],
                y=closed(upper),
                yaxis="y2",
                mode="lines",
                line={"width": 0, "shape": "hv"},
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            Scatter(
                x=[*hours, 24],
                y=closed(lower),
                yaxis="y2",
                mode="lines",
                line={"width": 0, "shape": "hv"},
                fill="tonexty",
                fillcolor=PRICE_BAND,
                name="Price 5–95th pct",
                legendrank=6,
                hoverinfo="skip",
            )
        )
    # Like the SoC mean line, the price line carries its band's values in the
    # unified hover (the band traces themselves stay hover-silent).
    if lower is not None and upper is not None:
        customdata = list(zip(closed(lower), closed(upper), strict=True))
        hovertemplate = (
            "%{y:.1f} p/kWh mean · 5–95th: %{customdata[0]:.1f}–%{customdata[1]:.1f}"
            "<extra>Price</extra>"
        )
    else:
        customdata = None
        hovertemplate = "%{y:.1f} p/kWh<extra>Price</extra>"
    fig.add_trace(
        Scatter(
            x=[*hours, 24],
            y=closed(price_values),
            yaxis="y2",
            mode="lines",
            line={"width": 2, "color": PRICE_COLOR, "shape": "hv"},
            # With a spread band the wash carries the story; without one the
            # fill-to-zero keeps the panel's original look.
            fill=None if price_band is not None else "tozeroy",
            fillcolor=None if price_band is not None else PRICE_FILL,
            name=f"Electricity price ({price_source or 'profile'})",
            legendrank=5,
            customdata=customdata,
            hovertemplate=hovertemplate,
        )
    )


def _style(fig: Figure, with_prices: bool) -> None:
    tick_font = {"color": INK_MUTED, "size": 12}
    fig.update_layout(
        template="none",
        height=600 if with_prices else 500,
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font={"family": FONT_FAMILY, "color": INK_PRIMARY, "size": 13},
        hovermode="x unified",
        # One hover label spanning every panel on the shared x-axis.
        hoversubplots="axis",
        hoverlabel={
            "bgcolor": SURFACE,
            "bordercolor": GRIDLINE,
            "font": {"family": FONT_FAMILY, "size": 12},
        },
        legend={
            "orientation": "h",
            "traceorder": "normal",
            "yanchor": "bottom",
            "y": 1.03,
            "xanchor": "right",
            "x": 1,
            "font": {"size": 12, "color": INK_SECONDARY},
        },
        margin={"l": 56, "r": 36, "t": 56, "b": 40},
        bargap=0,
        barcornerradius=3,
        xaxis={
            "range": [0, 24],
            "tickvals": list(range(0, 25, 3)),
            "ticktext": [f"{h:02d}:00" for h in range(0, 25, 3)],
            "showgrid": False,
            "linecolor": BASELINE,
            "ticks": "outside",
            "tickcolor": BASELINE,
            "tickfont": tick_font,
            "anchor": "y2" if with_prices else "y",
        },
        yaxis={
            "domain": [0.32, 1.0] if with_prices else [0.0, 1.0],
            "range": [0, 101],
            "ticksuffix": "%",
            "gridcolor": GRIDLINE,
            "zeroline": False,
            "tickfont": tick_font,
        },
    )
    if with_prices:
        fig.update_layout(
            yaxis2={
                "domain": [0.0, 0.24],
                "title": {"text": "p/kWh", "font": {"color": INK_MUTED, "size": 12}},
                "rangemode": "tozero",
                "gridcolor": GRIDLINE,
                "zeroline": False,
                "tickfont": tick_font,
                "anchor": "x",
            }
        )


def build_agent_chart(
    result: SimulationResult,
    agent_index: int,
    n_days: int | None = None,
    price_values: list[float] | None = None,
    price_source: str | None = None,
) -> Figure:
    """One driver's SoC trajectory and plug-in sessions over consecutive days.

    Spans the whole simulation after burn-in by default (`n_days` limits it),
    so the full behaviour — charging cadence, weekday/weekend rhythm — is
    visible at once; zoom in for detail. Plugged-in periods are shaded in the
    population chart's blue; each plug-in event is marked at the SoC it
    happened, since "SoC at plug-in" is a headline output of the simulator.
    When `price_values` is given — a full day-by-day series (one per simulated
    timestep) or a single-day profile (tiled) — the population chart's price
    panel is added below, showing each displayed day's prices.
    """
    config = result.config
    spd = config.steps_per_day
    step_hours = 24 / spd
    start_day = config.burn_in_days
    first = start_day * spd
    last = config.n_steps if n_days is None else min((start_day + n_days) * spd, config.n_steps)

    # A real time axis (anchored to an arbitrary Monday, matching the
    # simulation's dateless week) so ticks show day names and the hover
    # header reads "Thu 05:30". Only weekday and time are ever displayed.
    base = datetime(2024, 1, 1) + timedelta(days=start_day)
    times = [base + timedelta(hours=(step - first) * step_hours) for step in range(first, last)]
    soc = result.soc[agent_index, first:last] * 100
    plugged = result.plugged[agent_index, first:last]

    fig = Figure()
    fig.add_trace(
        Scatter(
            x=times,
            y=[100.0 if p else 0.0 for p in plugged],
            mode="lines",
            line={"width": 0, "shape": "hv"},
            fill="tozeroy",
            fillcolor=PLUGGED_IN_WASH,
            name="Plugged in",
            legendrank=1,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        Scatter(
            x=times,
            y=soc,
            mode="lines",
            line={"width": 2.5, "color": SOC_COLOR},
            name="State of charge",
            legendrank=2,
            hovertemplate="%{y:.1f}%<extra>State of charge</extra>",
        )
    )

    in_window = (
        (result.plug_event_agent == agent_index)
        & (result.plug_event_step >= first)
        & (result.plug_event_step < last)
    )
    fig.add_trace(
        Scatter(
            x=[
                base + timedelta(hours=(int(step) - first) * step_hours)
                for step in result.plug_event_step[in_window]
            ],
            y=result.plug_event_soc[in_window] * 100,
            mode="markers",
            marker={"size": 9, "color": SOC_COLOR, "line": {"width": 2, "color": SURFACE}},
            name="Plug-in event",
            legendrank=3,
            hovertemplate="plugged in at %{y:.1f}%<extra>Plug-in event</extra>",
        )
    )

    # Dotted reference at this driver's charging target: not data, not grid.
    target = result.agents[agent_index].target_soc * 100
    window_end = base + timedelta(hours=(last - first) * step_hours)
    fig.add_shape(
        type="line",
        x0=base,
        x1=window_end,
        y0=target,
        y1=target,
        line={"width": 1, "color": BASELINE, "dash": "dot"},
    )
    fig.add_annotation(
        x=window_end,
        y=target,
        text=f"target {target:.0f}%",
        showarrow=False,
        font={**ANNOTATION_FONT, "size": 11, "color": INK_MUTED},
        xanchor="right",
        yanchor="bottom",
        yshift=2,
    )

    with_prices = price_values is not None
    if with_prices:
        window = _window_prices(price_values, first, last, spd)
        _add_agent_price_panel(fig, times, step_hours, window, price_source)

    # Range buttons jump to spans anchored at the start of the data. Plotly's
    # built-in rangeselector steps backward from the current view's end, so on
    # the first day "1w" would reach six days before the data and show an
    # almost-empty chart. Spans that wouldn't differ from "All" are dropped.
    total_days = (last - first) // spd
    range_buttons = [
        {
            "label": label,
            "method": "relayout",
            "args": [{"xaxis.range": [base, base + timedelta(days=days)]}],
        }
        for days, label in ((1, "1d"), (3, "3d"), (7, "1w"), (14, "2w"))
        if days < total_days
    ]
    range_buttons.append(
        {"label": "All", "method": "relayout", "args": [{"xaxis.range": [base, window_end]}]}
    )

    fig.update_layout(
        template="none",
        height=540 if with_prices else 420,
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font={"family": FONT_FAMILY, "color": INK_PRIMARY, "size": 13},
        hovermode="x unified",
        # One hover label spanning every panel on the shared x-axis.
        hoversubplots="axis",
        hoverlabel={
            "bgcolor": SURFACE,
            "bordercolor": GRIDLINE,
            "font": {"family": FONT_FAMILY, "size": 12},
        },
        legend={
            "orientation": "h",
            "traceorder": "normal",
            "yanchor": "bottom",
            "y": 1.06,
            "xanchor": "right",
            "x": 1,
            "font": {"size": 12, "color": INK_SECONDARY},
        },
        margin={"l": 56, "r": 36, "t": 56, "b": 24},
        updatemenus=[
            {
                "type": "buttons",
                "direction": "right",
                "buttons": range_buttons,
                "x": 0,
                "xanchor": "left",
                "y": 1.06,
                "yanchor": "bottom",
                "showactive": True,
                "bgcolor": SURFACE,
                "bordercolor": GRIDLINE,
                "borderwidth": 1,
                "font": {"size": 11, "color": INK_SECONDARY},
            }
        ],
        xaxis={
            # Open on a single day; the range buttons and slider reach the rest.
            "range": [base, base + timedelta(days=1)],
            "rangeslider": {
                "visible": True,
                "thickness": 0.09,
                "bgcolor": SURFACE,
                "bordercolor": GRIDLINE,
                "borderwidth": 1,
            },
            # Two-line hour + day-name labels when zoomed in, day names alone
            # when zoomed out — the anchor date is arbitrary and never shown.
            "tickformatstops": [
                {"dtickrange": [None, 86_400_000], "value": "%H:%M<br>%a"},
                {"dtickrange": [86_400_000, None], "value": "%a"},
            ],
            "hoverformat": "%a %H:%M",
            "showgrid": True,
            "gridcolor": GRIDLINE,
            "linecolor": BASELINE,
            "ticks": "outside",
            "tickcolor": BASELINE,
            "tickfont": {"color": INK_MUTED, "size": 12},
            "anchor": "y2" if with_prices else "y",
        },
        yaxis={
            "domain": [0.32, 1.0] if with_prices else [0.0, 1.0],
            "range": [0, 101],
            "ticksuffix": "%",
            "gridcolor": GRIDLINE,
            "zeroline": False,
            "tickfont": {"color": INK_MUTED, "size": 12},
        },
    )
    if with_prices:
        fig.update_layout(
            yaxis2={
                "domain": [0.0, 0.24],
                "title": {"text": "p/kWh", "font": {"color": INK_MUTED, "size": 12}},
                "rangemode": "tozero",
                "gridcolor": GRIDLINE,
                "zeroline": False,
                "tickfont": {"color": INK_MUTED, "size": 12},
                "anchor": "x",
            }
        )
    return fig


def _window_prices(
    price_values: list[float], first: int, last: int, steps_per_day: int
) -> list[float]:
    """The price at each displayed step: slice a full day-by-day series, or
    tile a single-day profile."""
    if len(price_values) == steps_per_day:
        return [price_values[step % steps_per_day] for step in range(first, last)]
    return list(price_values[first:last])


def _add_agent_price_panel(
    fig: Figure,
    times: list[datetime],
    step_hours: float,
    window_prices: list[float],
    price_source: str | None,
) -> None:
    """The population chart's price panel, showing each displayed day's prices."""
    # Repeat the last value one step past the window so the step line spans
    # the same x extent as the SoC panel.
    fig.add_trace(
        Scatter(
            x=[*times, times[-1] + timedelta(hours=step_hours)],
            y=[*window_prices, window_prices[-1]],
            yaxis="y2",
            mode="lines",
            line={"width": 2, "color": PRICE_COLOR, "shape": "hv"},
            fill="tozeroy",
            fillcolor=PRICE_FILL,
            name=f"Electricity price ({price_source or 'profile'})",
            legendrank=4,
            hovertemplate="%{y:.1f} p/kWh<extra>Price</extra>",
        )
    )
