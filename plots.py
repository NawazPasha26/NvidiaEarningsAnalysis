"""
plots.py
Author: Nawaz Pasha

All plot-building utilities used across the dashboard:
- Time-series plots with earnings markers
- Decomposed multi-series plots
- Event-line overlays
- Progressive and interval charts
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from constants import COLORS
from analysis import get_next_trading_day


# =====================================================================
# UNIVERSAL LAYOUT HELPER
# =====================================================================

def apply_plotly_layout(
    fig,
    title="",
    ytitle=None,
    xtitle=None,
    bottom=80,
    top=56,
    legend_bottom=True
):
    """
    Applies a consistent layout across all visualizations.
    Ensures legends do not overlap graphs and axes are readable.
    """

    if legend_bottom:
        legend_dict = dict(
            orientation="h",
            x=0.0, xanchor="left",
            y=-0.18, yanchor="top",
            traceorder="normal",
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(0,0,0,0)"
        )
        margin_bottom = max(bottom, 140)
    else:
        legend_dict = dict(
            orientation="h",
            x=0.0, xanchor="left",
            y=1.02, yanchor="bottom",
            traceorder="normal",
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(0,0,0,0)"
        )
        margin_bottom = bottom

    fig.update_layout(
        title=dict(text=title, x=0.01, y=0.96, xanchor="left"),
        autosize=True,
        margin=dict(l=10, r=10, t=top, b=margin_bottom),
        legend=legend_dict
    )

    if ytitle:
        fig.update_yaxes(title_text=ytitle)
    if xtitle:
        fig.update_xaxes(title_text=xtitle)

    return fig


# =====================================================================
# EARNINGS OVERLAY HELPERS
# =====================================================================

def _ensure_series(x):
    return x.reset_index(drop=True) if isinstance(x, pd.Series) else pd.Series(x)


def get_event_reaction_days(date_series, earnings_dates):
    """
    Converts earnings dates → actual market reaction dates using next trading day logic.
    """
    d = _ensure_series(date_series)
    out = []

    for ed in earnings_dates.dropna():
        d0 = get_next_trading_day(d, pd.Timestamp(ed))
        if d0 is not None:
            out.append(pd.Timestamp(d0))

    return sorted(set(out))


def add_earnings_day_lines(fig, x_dates, earnings_dates, color="red"):
    """
    Adds vertical dashed lines on earnings reaction days.
    """
    ev_days = get_event_reaction_days(x_dates, earnings_dates)
    if not ev_days:
        return fig

    shapes = list(fig.layout.shapes) if fig.layout.shapes else []

    for d in ev_days:
        shapes.append(dict(
            type="line",
            xref="x", yref="paper",
            x0=d, x1=d,
            y0=0, y1=1,
            line=dict(color=color, width=1, dash="dot")
        ))

    fig.update_layout(shapes=shapes)
    return fig


def add_earnings_day_markers(
    fig, x_dates, y_values, earnings_dates,
    color="red", name="Earnings (Day 0)"
):
    """
    Adds explicit 'X' markers on earnings reaction days.
    """
    xd, ys = _ensure_series(x_dates), _ensure_series(y_values)
    xs_idx = {pd.Timestamp(x): i for i, x in enumerate(xd)}

    xs, ys_plot, text = [], [], []

    for d in get_event_reaction_days(xd, earnings_dates):
        i = xs_idx.get(pd.Timestamp(d))
        if i is None:
            continue
        if i >= len(ys) or pd.isna(ys[i]):
            continue

        xs.append(d)
        ys_plot.append(ys[i])
        text.append(
            f"Earnings Reaction<br>{d.strftime('%Y-%m-%d')}<br>Value: {ys[i]:.4%}"
        )

    if xs:
        fig.add_trace(go.Scatter(
            x=xs, y=ys_plot,
            mode="markers",
            name=name,
            marker=dict(color=color, symbol="x", size=7),
            hovertemplate="%{text}<extra></extra>",
            text=text,
            showlegend=False
        ))

    return fig


# =====================================================================
# MAIN PLOT FUNCTIONS
# =====================================================================

def plot_timeseries_with_event_lines(
    x, y, name, color, title, ytitle, earnings,
    hover="%{x|%Y-%m-%d}<br>%{y:.3%}<extra></extra>"
):
    """
    Standard timeseries plot with earnings overlays.
    """
    fig = go.Figure(go.Scatter(
        x=x, y=y,
        mode="lines",
        name=name,
        line=dict(color=color),
        hovertemplate=hover
    ))

    fig = add_earnings_day_lines(fig, x, earnings, COLORS["red"])
    fig = add_earnings_day_markers(fig, x, y, earnings, COLORS["red"])

    return apply_plotly_layout(fig, title=title, ytitle=ytitle)


def plot_multitimeseries_with_event_lines(
    x, series, earnings, title, ytitle
):
    """
    Multi-line decomposed return plots with event markers.
    """
    fig = go.Figure()

    for name, y, width, color in series:
        fig.add_trace(go.Scatter(
            x=x, y=y,
            name=name,
            mode="lines",
            line=dict(width=width, color=color),
            hovertemplate="<b>%{meta}</b><br>%{x|%Y-%m-%d}<br>%{y:.3%}<extra></extra>",
            meta=name
        ))

    fig = add_earnings_day_lines(fig, x, earnings)
    if series:
        fig = add_earnings_day_markers(fig, x, series[0][1], earnings)

    return apply_plotly_layout(fig, title=title, ytitle=ytitle)


# =====================================================================
# EVENT-ALIGNED PLOTS
# =====================================================================

def plot_avg_returns_selected(idxs, avg_dict, selected):
    """
    Plots mean event-aligned returns (daily).
    """
    fig = go.Figure()

    mapping = {
        "Total": ("total", "Total (NVIDIA)", COLORS["blue"]),
        "Systematic": ("factor", "Systematic (Factors)", COLORS["orange"]),
        "Idiosyncratic": ("idio", "Idiosyncratic", COLORS["green"]),
    }

    for label in selected:
        if label not in mapping:
            continue
        key, disp, col = mapping[label]

        fig.add_trace(go.Scatter(
            x=idxs, y=avg_dict[key],
            name=disp,
            mode="lines",
            line=dict(color=col),
            hovertemplate="Day %{x}: %{y:.3%}<extra></extra>"
        ))

    fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="red")

    return apply_plotly_layout(
        fig,
        title="Average Daily Returns Around Earnings",
        xtitle="Event Day (0 = Reaction Day)",
        ytitle="Average Return",
        bottom=90
    )


def plot_cum_event_selected(idxs, avg_dict, selected):
    """
    Plots compounded cumulative returns from avg_dict.
    -> NOW QUANT-CORRECT (compounded), matches R1 logic.
    """
    fig = go.Figure()

    mapping = {
        "Total": ("total", "Cumulative Total", COLORS["blue"]),
        "Systematic": ("factor", "Cumulative Systematic", COLORS["orange"]),
        "Idiosyncratic": ("idio", "Cumulative Idiosyncratic", COLORS["green"]),
    }

    for label in selected:
        if label not in mapping:
            continue

        key, disp, col = mapping[label]
        x = avg_dict[key]

        # Compound cumulative return correctly
        cum = (1 + x).cumprod() - 1

        fig.add_trace(go.Scatter(
            x=idxs, y=cum,
            name=disp,
            mode="lines",
            line=dict(color=col),
            hovertemplate="Day %{x}: %{y:.3%}<extra></extra>"
        ))

    fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="red")

    return apply_plotly_layout(
        fig,
        title="Cumulative Average Returns (Compounded)",
        xtitle="Event Day",
        ytitle="Cumulative Return",
        bottom=90
    )
