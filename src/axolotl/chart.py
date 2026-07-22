"""The population chart: % plugged in and state of charge over a typical day.

Both series are percentages, so they honestly share a single 0-100% axis —
no dual-axis tricks. Electricity prices are a different unit (p/kWh) and get
their own slim panel below, aligned on the same time axis.

Design notes: the bars are deliberately translucent so the state-of-charge
story reads first; the one direct label (the cheapest slot) is sparse by
intent and everything else lives in the unified hover and the legend. The
shaded window spanning both panels marks the priciest contiguous hours of the
day — the grid peak flexible charging exists to avoid.
"""

import polars as pl
from plotly.graph_objects import Bar, Figure, Scatter

# Categorical palette slots (fixed assignment: the entity keeps its hue).
PLUGGED_IN_COLOR = "rgba(42, 120, 214, 0.45)"  # blue, quiet
SOC_COLOR = "#eb6834"  # orange
SOC_BAND_OUTER = "rgba(235, 104, 52, 0.10)"  # 5-95th percentile wash
SOC_BAND_INNER = "rgba(235, 104, 52, 0.16)"  # 25-75th percentile wash
PRICE_COLOR = "#1baf7a"  # aqua
PRICE_FILL = "rgba(27, 175, 122, 0.08)"

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
PEAK_WASH = "rgba(11, 11, 11, 0.06)"

# Width of the highlighted peak window; its position is derived from the
# price profile. The 17:00-20:00 fallback is GB's documented evening grid peak,
# used when the chart is built without prices.
PEAK_WINDOW_HOURS = 3
DEFAULT_GRID_PEAK = (17.0, 20.0)

FONT_FAMILY = 'system-ui, -apple-system, "Segoe UI", sans-serif'
ANNOTATION_FONT = {"family": FONT_FAMILY, "size": 12, "color": INK_SECONDARY}


def build_population_chart(
    profile: pl.DataFrame,
    price_values: list[float] | None = None,
    price_source: str | None = None,
) -> Figure:
    """Build the dashboard figure from a time-of-day profile.

    `profile` is the output of `aggregate.time_of_day_profile`. When
    `price_values` is given (one per timestep), a price panel is added below.
    """
    hours = profile["hour"].to_list()
    step = hours[1] - hours[0] if len(hours) > 1 else 0.5
    centers = [h + step / 2 for h in hours]

    with_prices = price_values is not None
    # Both panels live on ONE x-axis as stacked y-domains (rather than
    # make_subplots' separate axes) so `hoversubplots="axis"` can show the
    # unified hover across the SoC panel and the price panel together.
    fig = Figure()

    _add_peak_window(fig, price_values, with_prices)
    _add_plugged_in_bars(fig, centers, profile, step)
    _add_soc_layers(fig, centers, profile)
    if with_prices:
        _add_price_panel(fig, hours, price_values, price_source)
        _annotate_cheapest_slot(fig, centers, price_values)
    _style(fig, with_prices)
    return fig


def _priciest_window(price_values: list[float]) -> tuple[float, float]:
    """Start/end hours of the most expensive contiguous PEAK_WINDOW_HOURS."""
    step_hours = 24 / len(price_values)
    window = round(PEAK_WINDOW_HOURS / step_hours)
    start = max(
        range(len(price_values) - window + 1),
        key=lambda i: sum(price_values[i : i + window]),
    )
    return start * step_hours, (start + window) * step_hours


def _add_peak_window(fig: Figure, price_values: list[float] | None, with_prices: bool) -> None:
    """Shade the priciest hours of the day across every panel."""
    if price_values is not None:
        peak_start, peak_end = _priciest_window(price_values)
        label = f"priciest {PEAK_WINDOW_HOURS} hours"
    else:
        peak_start, peak_end = DEFAULT_GRID_PEAK
        label = "evening grid peak"

    # add_shape rather than add_vrect: the latter drops the shape silently
    # under plotly 6.9. `yref="y domain"` spans a panel's full height.
    for yref in ("y domain", "y2 domain") if with_prices else ("y domain",):
        fig.add_shape(
            type="rect",
            x0=peak_start,
            x1=peak_end,
            y0=0,
            y1=1,
            xref="x",
            yref=yref,
            fillcolor=PEAK_WASH,
            line_width=0,
            layer="below",
        )
    fig.add_annotation(
        x=(peak_start + peak_end) / 2,
        y=99,
        text=label,
        showarrow=False,
        font={**ANNOTATION_FONT, "size": 11, "color": INK_MUTED},
        yanchor="top",
    )


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
) -> None:
    # Repeat the last value at 24:00 so the step line spans the full day and
    # both panels share an identical x extent.
    fig.add_trace(
        Scatter(
            x=[*hours, 24],
            y=[*price_values, price_values[-1]],
            yaxis="y2",
            mode="lines",
            line={"width": 2, "color": PRICE_COLOR, "shape": "hv"},
            fill="tozeroy",
            fillcolor=PRICE_FILL,
            name=f"Electricity price ({price_source or 'profile'})",
            legendrank=5,
            hovertemplate="%{y:.1f} p/kWh<extra>Price</extra>",
        )
    )


def _annotate_cheapest_slot(fig: Figure, centers: list[float], price_values: list[float]) -> None:
    """The one direct label: where charging is cheapest."""
    cheapest_idx = min(range(len(price_values)), key=price_values.__getitem__)
    fig.add_annotation(
        x=centers[cheapest_idx],
        y=price_values[cheapest_idx],
        xref="x",
        yref="y2",
        text=f"cheapest: {price_values[cheapest_idx]:.1f}p",
        showarrow=False,
        font={**ANNOTATION_FONT, "size": 11},
        yanchor="bottom",
        yshift=6,
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
