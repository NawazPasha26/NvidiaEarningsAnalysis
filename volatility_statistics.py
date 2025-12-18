# tabs/volatility_statistics.py

"""
Event Volatility & Statistical Reliability Tab
Author : Nawaz Pasha
---------------------------------------------

This module renders the **Event Volatility & Statistical Reliability** tab.

Purpose:
- Analyze volatility behavior around earnings events
- Compare rolling volatility and event-aligned volatility
- Measure volatility uplift pre vs post earnings
- Visualize volatility heatmaps and EWMA-smoothed volatility curves
- Perform bootstrap-based statistical significance tests
- Evaluate reliability of earnings-related volatility and returns
- Provide hypothesis-driven interpretation of results

Inputs:
- df      : Daily return dataframe
- df_e    : Earnings dates dataframe
- ev_out  : Output of build_event_window (mats, idxs, event_info)
- win     : Event window size (± days)
- roll    : Rolling window length for volatility

Outputs:
- Streamlit-rendered charts, tables, KPIs
- No mutation of upstream data (presentation layer only)

Design Notes:
- Numerical/statistical routines are sourced from analysis.py
- Plot layout helpers are sourced from plots.py
- UI helpers are defined locally to preserve exact rendering
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from constants import COLORS
from analysis import (
    compute_event_averages,
    mask_triplet,
    compute_pre_post_abs_change,
    bootstrap_vol_tests,
)
from plots import apply_plotly_layout


def render_volatility_statistics(df, df_e, ev_out, win, roll):
    """
    Renders the Event Volatility & Statistical Reliability tab.

    Called from app.py inside the Tab-3 context.
    """

   # ---------------------------------------------------------------
    # MAIN TITLE
    # ---------------------------------------------------------------
    st.markdown(
        "<h2 style='font-size:22px; font-weight:650; margin-bottom:4px;'>"
        "📉 Event Volatility & Statistical Reliability"
        "</h2>",
        unsafe_allow_html=True
    )

    ann = np.sqrt(252)
    st.markdown("<div style='margin-top:2px;'></div>", unsafe_allow_html=True)

    # ---------------------------------------------------------------
    # LOAD EVENT OBJECTS (CORRECT / CONSISTENT)
    # ---------------------------------------------------------------
    mats, idxs, event_info = ev_out
    idxs2, avg = compute_event_averages(ev_out)
    pre_mask, day0_mask, post_mask = mask_triplet(idxs2)

    # =====================================================================
    # VOLATILITY AROUND EARNINGS — ROLLING VOL & ABS CHANGE
    # =====================================================================
    st.markdown(
        "<h3 style='font-size:18px; font-weight:600; margin-bottom:4px;'>"
        "Volatility Around Earnings"
        "</h3>",
        unsafe_allow_html=True
    )

    # Rolling volatility
    df["NVDA_RollVol"] = df["NVDA"].rolling(roll).std() * ann
    df["Idio_RollVol"] = df["Idio_Return"].rolling(roll).std() * ann

    # Pre/Post change
    pre_abs, post_abs, change = compute_pre_post_abs_change(mats, idxs)

    vol_df = pd.DataFrame({
        "EventDate": event_info["event_date"],
        "PrevAvgAbsRet": pre_abs,
        "PostAvgAbsRet": post_abs,
        "Change": change
    }).sort_values("EventDate")

    col1, col2 = st.columns(2, gap="small")

    # --------------------- LEFT CHART — ROLLING VOL -----------------------
    with col1:
        st.markdown(
            "<h4 style='font-size:16px; font-weight:600; margin-bottom:2px;'>"
            "Rolling Volatility"
            "</h4>",
            unsafe_allow_html=True
        )
        st.markdown("<p style='font-size:12px; line-height:1.3;'>Shows how NVIDIA’s volatility evolved over time, with earnings dates marked to highlight volatility spikes.</p>", unsafe_allow_html=True)

        fig_roll = go.Figure()
        fig_roll.add_trace(go.Scatter(
            x=df["Date"], y=df["NVDA_RollVol"],
            name=f"NVDA {roll}D Vol", line=dict(color=COLORS["purple"], width=2)
        ))
        fig_roll.add_trace(go.Scatter(
            x=df["Date"], y=df["Idio_RollVol"],
            name=f"Idio {roll}D Vol", line=dict(color=COLORS["brown"], width=2)
        ))
        for d in df_e["EarningsDate"]:
            fig_roll.add_vline(x=d, line_color="gray", opacity=0.25)
        fig_roll = apply_plotly_layout(fig_roll, title="", ytitle="Volatility")
        fig_roll.update_layout(height=300, margin=dict(t=20, b=30))
        st.plotly_chart(fig_roll, use_container_width=True)


    # --------------------- RIGHT CHART — PRE/POST CHANGE -------------------
    with col2:
        st.markdown(
            "<h4 style='font-size:16px; font-weight:600; margin-bottom:2px;'>"
            "Change in Avg |Daily Return| (Post – Pre)"
            "</h4>",
            unsafe_allow_html=True
        )
        
        st.markdown("<p style='font-size:12px; line-height:1.3;'>Compares average daily moves before and after each earnings event to detect shifts in volatility.</p>", unsafe_allow_html=True)

        fig_change = go.Figure(go.Bar(
            x=vol_df["EventDate"].dt.strftime("%Y-%m-%d"),
            y=vol_df["Change"],
            marker_color=np.where(vol_df["Change"] > 0, COLORS["green"], COLORS["red"]),
            customdata=np.stack([vol_df["PrevAvgAbsRet"], vol_df["PostAvgAbsRet"]], axis=1),
            hovertemplate=(
                "Event: %{x}<br>"
                "Post – Pre: %{y:.4f}<br>"
                "PrevAvgAbsRet: %{customdata[0]:.4f}<br>"
                "PostAvgAbsRet: %{customdata[1]:.4f}<extra></extra>"
            )
        ))
        fig_change = apply_plotly_layout(fig_change, title="", ytitle="Δ Abs Return")
        fig_change.update_layout(height=300, margin=dict(t=20, b=30))
        st.plotly_chart(fig_change, use_container_width=True)


    st.markdown("<div style='margin-top:2px;'></div>", unsafe_allow_html=True)

    # =====================================================================
    # LEFT: KPIs + MULTI-WINDOW TABLE,
    # RIGHT: EVENT-LEVEL VOLATILITY TABLE
    # =====================================================================
    
    st.markdown(
        "<h3 style='font-size:17px; font-weight:600; margin-bottom:2px;'>"
        "Volatility Uplift & Cross-Window Comparison"
        "</h3>",
        unsafe_allow_html=True
    )
    
    # -------------------------------------------------------
    # PREPARE BOTH TABLES + KPI VALUES
    # -------------------------------------------------------
    
    # --- KPIs ---
    uplift_total = np.mean(post_abs - pre_abs)
    uplift_ratio = np.mean(post_abs / pre_abs - 1)
    frac_events_higher = np.mean(post_abs > pre_abs)
    
    def kpi_tile(label, value, icon):
        color = "green" if value > 0 else "red" if value < 0 else "#333"
        return f"""
        <div style='
            width:105px;
            height:58px;
            border-radius:6px;
            background:white;
            border:1px solid #ddd;
            display:flex;
            flex-direction:column;
            justify-content:center;
            align-items:center;
            padding:2px 0;
            margin:0;
            box-shadow:0 1px 2px rgba(0,0,0,0.04);
        '>
            <div style="font-size:13px; line-height:1;">{icon}</div>
            <div style="font-size:9px; color:#666; font-weight:600; line-height:1.1; text-align:center;">
                {label}
            </div>
            <div style="font-size:12.5px; font-weight:700; color:{color}; line-height:1;">
                {value:.2%}
            </div>
        </div>
        """


    
    # --- Multi-Window Volatility Table ---
    windows = [5, 10, 20, 30]
    rows = []
    
    for w in windows:
        series = df["NVDA"].rolling(w).std() * ann
        evt_vals = []
        for ed in df_e["EarningsDate"]:
            idx0 = df["Date"].searchsorted(ed)
            if idx0 < len(series):
                evt_vals.append(series.iloc[idx0])
        rows.append({
            "Window": f"{w}D",
            "Median Vol": np.nanmedian(evt_vals),
            "Mean Vol": np.nanmean(evt_vals),
            "Vol Std": np.nanstd(evt_vals),
        })
    
    multi_df = pd.DataFrame(rows)
    
    # --- Event-Level Volatility ---
    event_rows = []
    for i, ed in enumerate(event_info["event_date"]):
        event_rows.append({
            "Earnings Date": ed.strftime("%Y-%m-%d"),
            "Pre Vol": pre_abs[i],
            "Post Vol": post_abs[i],
        })
    event_df = pd.DataFrame(event_rows)
    
    # --- Coloring ---
    def color_num(v):
        if isinstance(v, (int, float)):
            return "color:green;" if v > 0 else "color:red;font-weight:600;" if v < 0 else ""
        return ""
    
    def fmt4(x):
        try:
            return f"{float(x):.4f}"
        except:
            return x
    
    # -------------------------------------------------------
    # LAYOUT — LEFT: KPIs → Multi-window table
    #          RIGHT: Event-level table
    # -------------------------------------------------------
    
    left_col, right_col = st.columns([1.2, 1.8])
    
    # ======================
    # LEFT COLUMN
    # ======================
    with left_col:
    
        # --- KPIs ---
        st.markdown("<h4 style='font-size:15px; margin-bottom:4px;'>Volatility Uplift KPIs</h4>", unsafe_allow_html=True)
    
        k1, k2, k3 = st.columns([1, 1, 1], gap="small")
        k1.markdown(kpi_tile("Avg Vol Uplift", uplift_total, "📈"), unsafe_allow_html=True)
        k2.markdown(kpi_tile("Relative Uplift", uplift_ratio, "📊"), unsafe_allow_html=True)
        k3.markdown(kpi_tile("% Higher Vol", frac_events_higher, "🔥"), unsafe_allow_html=True)
    
        st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)
    
        # --- Multi-window table ---
        st.markdown("<h4 style='font-size:15px; margin-bottom:4px;'>Multi-Window Volatility Comparison</h4>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:12px; line-height:1.3;'>Shows volatility across multiple rolling windows to compare short-term versus medium-term risk around earnings.</p>", unsafe_allow_html=True)

        style_multi = (
            multi_df.style
            .set_properties(subset=["Window"], **{"font-weight": "700"})
            .applymap(color_num, subset=["Median Vol", "Mean Vol", "Vol Std"])
            .format(fmt4)
            .set_table_styles([
                {"selector": "th", "props": "background:#e8e8e8;font-weight:bold;text-align:center;"},
                {"selector": "td", "props": "text-align:center;border:1px solid #ddd;font-size:12px;padding:4px;"}
            ])
        )
    
        st.dataframe(style_multi, use_container_width=True, hide_index=True)
    
    # ======================
    # RIGHT COLUMN
    # ======================
    with right_col:
    
        st.markdown("<h4 style='font-size:15px; margin-bottom:4px;'>Event-Level Volatility (Pre, Post)</h4>",
                    unsafe_allow_html=True)
        st.markdown("<p style='font-size:12px; line-height:1.25; margin-bottom:6px;'>Provides per-event volatility before and after earnings to identify which quarters saw the biggest changes.</p>", unsafe_allow_html=True)

        
        style_event = (
            event_df.style
            .set_properties(subset=["Earnings Date"], **{"font-weight": "700"})
            .applymap(color_num, subset=["Pre Vol", "Post Vol"])
            .format(fmt4)
            .set_table_styles([
                {"selector":"th","props":"background:#e8e8e8;font-weight:bold;text-align:center;"},
                {"selector":"td","props":"text-align:center;border:1px solid #ddd;font-size:12px;padding:4px;"}
            ])
        )
    
        st.dataframe(style_event, use_container_width=True, hide_index=True)

    # =====================================================================
    # HEATMAP — TOGGLE (Z-SCORE or RAW VOLATILITY)
    # =====================================================================

    st.markdown(
        "<h3 style='font-size:18px; font-weight:600; margin-bottom:4px;'>"
        "Volatility Heatmap"
        "</h3>",
        unsafe_allow_html=True
    )
    st.markdown("<p style='font-size:12px; line-height:1.25; margin-bottom:6px;'>Visualizes volatility patterns for D0 +-10 days across all earnings events to highlight periods of turbulence.</p>", unsafe_allow_html=True)


    mode = st.radio(
        "Heatmap Mode",
        ["Z-Score Normalized", "Raw Volatility"],
        horizontal=True
    )

    win_h = 10
    heat_vals = []
    labels = []

    for ed in event_info["event_date"]:
        idx0 = df["Date"].searchsorted(ed)
        if idx0 - win_h < 0 or idx0 + win_h >= len(df):
            continue

        seg = df["NVDA"].iloc[idx0 - win_h : idx0 + win_h + 1]

        if mode == "Z-Score Normalized":
            seg_vals = (seg - seg.mean()) / seg.std()
        else:
            seg_vals = seg.rolling(2).std().fillna(0).values

        heat_vals.append(seg_vals)
        labels.append(ed.strftime("%Y-%m-%d"))

    heat_arr = np.array(heat_vals)

    fig_h = go.Figure(
        data=go.Heatmap(
            z=heat_arr,
            x=[f"D{d:+d}" for d in range(-win_h, win_h + 1)],
            y=labels,
            colorscale="RdBu",
            colorbar=dict(title="Volatility")
        )
    )

    fig_h = apply_plotly_layout(fig_h, title="")
    st.plotly_chart(fig_h, use_container_width=True)

    #st.markdown("<div style='margin-top:2px;'></div>", unsafe_allow_html=True)
    st.markdown("<div style='height:4px;'></div>", unsafe_allow_html=True)
    # =====================================================================
    # EWMA VOLATILITY AROUND EARNINGS
    # =====================================================================
    
    st.markdown("<h3 style='font-size:17px; margin-top:4px;'>📉 EWMA Volatility Around Earnings</h3>",
                unsafe_allow_html=True)
    
    with st.expander("📘 EWMA Volatility Curve, KPIs & Burst Classification", expanded=True):
    
        # ------------------------------------------------------------------
        # 1) COMPUTE EVENT-ALIGNED VOLATILITY MATRICES
        # ------------------------------------------------------------------
        # We reuse the event matrices from ev_out
        # mats: aligned returns for each event
        # idxs: index offsets (-win to +win)
        # We'll compute volatility per day across events
    
        # daily cross-sectional volatility for each relative day
        vol_curve = mats["NVDA"].std(axis=0)        # raw cross-sectional std
        days = np.arange(-win, win + 1)
    
        # ------------------------------------------------------------------
        # 2) EWMA VOLATILITY SMOOTHING (λ = 0.94 — RiskMetrics standard)
        # ------------------------------------------------------------------
        lambda_vol = 0.94
        ewma_vol = np.zeros_like(vol_curve)
        ewma_vol[0] = vol_curve[0]
    
        for i in range(1, len(vol_curve)):
            ewma_vol[i] = (
                lambda_vol * ewma_vol[i-1] +
                (1 - lambda_vol) * vol_curve[i]
            )
    
        # ------------------------------------------------------------------
        # 3) COMPUTE VOLATILITY KPIs
        # ------------------------------------------------------------------
        pre_vol = ewma_vol[:win].mean() if win > 0 else ewma_vol[0]
        event_vol = ewma_vol[win]        # day 0
        post_vol = ewma_vol[win+1:].mean() if win+1 < len(ewma_vol) else ewma_vol[-1]
    
        burst_ratio = (event_vol / pre_vol) if pre_vol != 0 else np.nan
    
        # ------------------------------------------------------------------
        # 4) CLASSIFICATION BADGE FOR VOLATILITY BURST
        # ------------------------------------------------------------------
        def classify_vol(burst):
            if burst >= 3.0:
                return "🔴 Extreme Volatility Burst (≥ 3×)"
            if burst >= 2.0:
                return "🟠 Strong Volatility Burst (≥ 2×)"
            if burst >= 1.3:
                return "🟡 Mild Volatility Uptick"
            return "🟢 Stable / Low Earnings Volatility"
    
        vol_badge = classify_vol(burst_ratio)
    
        # ------------------------------------------------------------------
        # KPI TILE STYLE
        # ------------------------------------------------------------------
        def kpi_vol_tile(label, value, icon="📊"):
            if isinstance(value, (float, int)):
                col = "green" if value < 0.02 else "#000"  # vol KPIs not pos/neg
                formatted = f"{value:.4f}"
            else:
                col = "#333"
                formatted = value
    
            return f"""
            <div style="
                border-radius:8px;
                padding:6px 6px;
                background-color:white;
                border:1px solid #e0e0e0;
                width:155px;
                height:70px;
                display:flex;
                flex-direction:column;
                justify-content:center;
                align-items:center;
                box-shadow:0 1px 2px rgba(0,0,0,0.05);
            ">
                <div style="font-size:16px;">{icon}</div>
                <div style="font-size:11px; color:#666; font-weight:600; text-align:center;">
                    {label}
                </div>
                <div style="font-size:14px; font-weight:700; color:{col};">
                    {formatted}
                </div>
            </div>
            """    
        st.markdown("<h4 style='font-size:14px; margin-bottom:6px;'>🔹 EWMA Volatility KPIs</h4>", 
                    unsafe_allow_html=True)
    
        k1, k2, k3, k4 = st.columns(4)
    
        with k1:
            st.markdown(
                kpi_vol_tile("Pre-Earnings Vol", pre_vol, "⬅️"),
                unsafe_allow_html=True
            )
        with k2:
            st.markdown(
                kpi_vol_tile("Event-Day Vol", event_vol, "⭐"),
                unsafe_allow_html=True
            )
        with k3:
            st.markdown(
                kpi_vol_tile("Post-Earnings Vol", post_vol, "➡️"),
                unsafe_allow_html=True
            )
        with k4:
            st.markdown(
                kpi_vol_tile("Vol Burst Ratio", burst_ratio, "⚡"),
                unsafe_allow_html=True
            )
    
        # ------------------------------------------------------------------
        # BURST CLASSIFICATION BADGE
        # ------------------------------------------------------------------
        st.markdown(
            f"""
            <div style="
                border-radius:8px;
                padding:8px 12px;
                margin-top:8px;
                background-color:#fff;
                border:1px solid #e0e0e0;
                width:98%;
                text-align:center;
                font-size:13px;
                font-weight:700;
                box-shadow:0 1px 2px rgba(0,0,0,0.05);
            ">
                {vol_badge}
            </div>
            """,
            unsafe_allow_html=True
        )
    
        # ------------------------------------------------------------------
        # SIDE-BY-SIDE CHARTS
        # ------------------------------------------------------------------
        left, right = st.columns(2)
    
        # LEFT: Raw vs EWMA Volatility Curve
        with left:
            st.markdown(
                "<h4 style='font-size:13px; margin-bottom:0px;'>📈 EWMA Volatility Curve (Event-Aligned)</h4>",
                unsafe_allow_html=True
            )
            st.markdown("<p style='font-size:12px; line-height:1.25; margin-bottom:6px;'>Smooths event-aligned volatility to reveal structural changes and volatility bursts on earnings day.</p>", unsafe_allow_html=True)

    
            fig_vol = go.Figure()
            fig_vol.add_trace(go.Scatter(
                x=days, y=vol_curve,
                name="Raw Vol (Std Dev)",
                opacity=0.45,
                line=dict(width=1.3, color="gray")
            ))
            fig_vol.add_trace(go.Scatter(
                x=days, y=ewma_vol,
                name=f"EWMA Vol (λ={lambda_vol})",
                line=dict(width=3)
            ))
            fig_vol.add_vline(x=0, line_color="red", line_dash="dash")
    
            fig_vol = apply_plotly_layout(fig_vol, title="", ytitle="Volatility")
            fig_vol.update_layout(height=270)
    
            st.plotly_chart(fig_vol, use_container_width=True)
    
        # RIGHT: Volatility Burst Ratio Mini-Chart
        with right:
            st.markdown(
                "<h4 style='font-size:13px; margin-bottom:0px;'>📊 Event-Day Volatility Burst</h4>",
                unsafe_allow_html=True
            )
    
            fig_burst = go.Figure()
            fig_burst.add_trace(go.Bar(
                x=["Pre Vol", "Event Vol"],
                y=[pre_vol, event_vol],
                marker_color=["gray", "red"]
            ))
    
            fig_burst = apply_plotly_layout(fig_burst, title="", ytitle="Volatility")
            fig_burst.update_layout(height=270)
    
            st.plotly_chart(fig_burst, use_container_width=True)
    
    # End EWMA Volatility module
    st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)
    

    # =====================================================================
    # BOOTSTRAPPED STATISTICAL SIGNIFICANCE — VOLATILITY
    # =====================================================================

    st.markdown(
        "<h3 style='font-size:18px; font-weight:600; margin-bottom:4px;'>"
        "Bootstrapped Statistical Significance of Volatility"
        "</h3>",
        unsafe_allow_html=True
    )

    colX, colY = st.columns(2)

    with colX:
        bootstrap_samples = st.slider(
            "Bootstrap Samples", 5000, 50000, 20000, step=5000
        )

    with colY:
        ci_choice = st.select_slider(
            "Confidence Interval", ["90%", "95%", "99%"], value="95%"
        )

    ci_level = {"90%": 0.90, "95%": 0.95, "99%": 0.99}[ci_choice]

    # Run the official bootstrap engine used across the app
    results_total, results_idio = bootstrap_vol_tests(
        mats, idxs, ci_level=ci_level, n_boot=bootstrap_samples
    )

    # ---- Assemble combined bootstrapped results table ----
    def build_vol_table():
        rows = []

        for metric, entry in results_total.items():
            rows.append({
                "Type": "Total |NVDA|",
                "Metric": metric,
                "Observed": entry["observed"],
                "CI Low": entry["ci"][0],
                "CI High": entry["ci"][1],
                "p-value": entry["p"]
            })

        for metric, entry in results_idio.items():
            rows.append({
                "Type": "Idiosyncratic |Idio|",
                "Metric": metric,
                "Observed": entry["observed"],
                "CI Low": entry["ci"][0],
                "CI High": entry["ci"][1],
                "p-value": entry["p"]
            })

        return pd.DataFrame(rows)
    
    st.markdown("<p style='font-size:12px; line-height:1.25; margin-bottom:6px;'>Tests whether earnings-day volatility is statistically different from normal using a bootstrap-based confidence interval.</p>", unsafe_allow_html=True)

    
    vol_tab = build_vol_table()

    style_vol = (
        vol_tab.style
        .set_properties(subset=["Type"], **{"font-weight": "700"})
        .applymap(color_num, subset=["Observed", "CI Low", "CI High"])
        .format({
            "Observed": fmt4,
            "CI Low": fmt4,
            "CI High": fmt4,
            "p-value": fmt4
        })
        .set_table_styles([
            {"selector": "th",
             "props": "background:#e8e8e8;font-weight:bold;text-align:center;"},
            {"selector": "td",
             "props": "text-align:center;font-size:12px;border:1px solid #ddd;padding:4px;"}
        ])
    )

    st.dataframe(style_vol, use_container_width=True)

    st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)

    # =====================================================================
    # RETURNS-BASED BOOTSTRAP RELIABILITY
    # =====================================================================

    st.markdown(
        "<h3 style='font-size:18px; font-weight:600; margin-bottom:4px;'>"
        "Bootstrapped Statistical Significance (Returns-Based)"
        "</h3>",
        unsafe_allow_html=True
    )

    # -------- Winsorization (5%–95%) --------
    raw_returns = df["NVDA"].dropna()
    lower, upper = np.percentile(raw_returns, [5, 95])
    cleaned_returns = np.clip(raw_returns, lower, upper)

    st.markdown(
        f"<p style='font-size:13px;'>Winsorization Applied → "
        f"<b>[{lower:.4%}, {upper:.4%}]</b></p>",
        unsafe_allow_html=True
    )

    # -------- Event windows --------
    pre_days = 10
    post_days = 10

    dates = df["Date"].reset_index(drop=True)
    returns = cleaned_returns.reset_index(drop=True)

    day0_vals, pre_vals, post_vals = [], [], []

    for ed in df_e["EarningsDate"]:
        idx0 = dates.searchsorted(ed, side="right")
        if idx0 >= len(returns):
            continue

        # Day-0
        day0_vals.append(returns.iloc[idx0])
        # Pre-window
        pre_vals.extend(returns.iloc[max(0, idx0-pre_days): idx0])
        # Post-window
        post_vals.extend(returns.iloc[idx0+1: idx0+1+post_days])

    day0_vals = np.array(day0_vals)
    pre_vals = np.array(pre_vals)
    post_vals = np.array(post_vals)

    # -------- Core bootstrap engine --------
    def bootstrap_ci_p(data, stat_fn, observed, n_boot=20000, ci=0.95):
        arr = np.array(data)
        idx = np.random.randint(0, len(arr), size=(n_boot, len(arr)))
        samples = arr[idx]

        # Compute stat on all bootstrap samples
        try:
            stats = stat_fn(samples, axis=1)
        except:
            stats = np.array([stat_fn(s) for s in samples])

        alpha = 1 - ci
        lo = np.percentile(stats, alpha/2 * 100)
        hi = np.percentile(stats, (1 - alpha/2) * 100)

        diffs = np.abs(stats - stats.mean())
        obs = np.abs(observed - stats.mean())
        p = np.mean(diffs >= obs)

        return (lo, hi), p

    # -------- Metrics to test --------
    tests = {
        "Mean Return": np.mean,
        "Median Return": np.median,
        "Volatility (Std Dev)": np.std,
        "% Positive Days": lambda x: np.mean(x > 0),
    }

    rows = []

    for name, fn in tests.items():

        # Day-0 significance
        obs_d0 = fn(day0_vals)
        ci_d0, p_d0 = bootstrap_ci_p(
            cleaned_returns,
            fn,
            obs_d0,
            bootstrap_samples,
            ci_level
        )

        # Pre vs Post change
        if len(pre_vals) > 0 and len(post_vals) > 0:
            m = min(len(pre_vals), len(post_vals))
            A = pre_vals[:m]
            B = post_vals[:m]

            obs_diff = fn(B) - fn(A)

            diffs = []
            for _ in range(bootstrap_samples):
                bsA = np.random.choice(A, m, replace=True)
                bsB = np.random.choice(B, m, replace=True)
                diffs.append(fn(bsB) - fn(bsA))

            diffs = np.array(diffs)

            lo, hi = np.percentile(diffs, [(1-ci_level)*50, 100-(1-ci_level)*50])
            diff_ci = (lo, hi)

            diff_p = np.mean(
                np.abs(diffs - diffs.mean()) >=
                np.abs(obs_diff - diffs.mean())
            )
        else:
            obs_diff = np.nan
            diff_ci = (np.nan, np.nan)
            diff_p = np.nan

        rows.append([
            name,
            obs_d0,
            f"[{ci_d0[0]:.4f}, {ci_d0[1]:.4f}]",
            p_d0,
            obs_diff,
            f"[{diff_ci[0]:.4f}, {diff_ci[1]:.4f}]",
            diff_p,
        ])
        
    st.markdown("<p style='font-size:12px; line-height:1.25; margin-bottom:6px;'>Evaluates whether earnings-day returns and before-vs-after changes are statistically significant or simply noise.</p>", unsafe_allow_html=True)

    res_df = pd.DataFrame(rows, columns=[
        "Statistic",
        "Day-0 Value",
        f"Day-0 CI ({ci_choice})",
        "Day-0 p-value",
        "Pre–Post Δ",
        f"Δ CI ({ci_choice})",
        "Δ p-value",
    ])

    # -------- Style the results table --------
    style_res = (
        res_df.style
        .set_properties(subset=["Statistic"], **{"font-weight": "700"})
        .applymap(color_num, subset=["Day-0 Value", "Pre–Post Δ"])
        .set_table_styles([
            {"selector": "th",
             "props": "background:#e8e8e8;font-weight:bold;text-align:center;"},
            {"selector": "td",
             "props": "text-align:center;border:1px solid #ddd;font-size:12px;padding:4px;"}
        ])
    )

    st.dataframe(style_res, use_container_width=True)

    st.markdown("<div style='margin-top:8px;'></div>", unsafe_allow_html=True)

    # =====================================================================
    # INTERPRETATION — OVERALL HYPOTHESIS TESTING CONCLUSIONS
    # =====================================================================

    with st.expander("🧠 Overall Hypothesis Testing Conclusions", expanded=True):

        st.markdown(
            "<h3 style='font-size:18px; font-weight:600; margin-bottom:6px;'>"
            "📌 Statistical significance Test Findings"
            "</h3>",
            unsafe_allow_html=True
        )

        # ---------------------------
        # MEAN RETURN INTERPRETATION
        # ---------------------------
        mean_row = res_df.loc[res_df["Statistic"] == "Mean Return"].iloc[0]
        mean_p = mean_row["Day-0 p-value"]
        mean_ci = mean_row[f"Day-0 CI ({ci_choice})"]

        if mean_p < 0.05:
            st.markdown(f"""
            **Mean Return:**  
            • Earnings day produces a **statistically significant abnormal return**.  
            • Confidence Interval: **{mean_ci}**  
            → Indicates that earnings announcements often convey new information that
              materially shifts NVIDIA’s price.
            """)
        else:
            st.markdown(f"""
            **Mean Return:**  
            • No statistically reliable abnormal return on earnings day.  
            • Confidence Interval: **{mean_ci}**  
            → Suggests that while some earnings days have large jumps, the *average*
              effect is not consistently different from a normal trading day.
            """)

        # ---------------------------
        # MEDIAN RETURN INTERPRETATION
        # ---------------------------
        med_row = res_df.loc[res_df["Statistic"] == "Median Return"].iloc[0]
        med_p = med_row["Day-0 p-value"]
        med_ci = med_row[f"Day-0 CI ({ci_choice})"]

        if med_p < 0.05:
            st.markdown(f"""
            **Median Return:**  
            • The *typical* earnings-day return differs significantly from normal days.  
            • CI: **{med_ci}**  
            → Indicates broad, consistent shifts rather than extreme outliers.
            """)
        else:
            st.markdown(f"""
            **Median Return:**  
            • Median return shows **no significant deviation** from normal days.  
            • CI: **{med_ci}**  
            → Implies that most earnings reactions are mild; large moves in a few events
              are primarily responsible for any mean effects.
            """)

        # ---------------------------
        # VOLATILITY RETURN INTERPRETATION
        # ---------------------------
        vol_row = res_df.loc[res_df["Statistic"] == "Volatility (Std Dev)"].iloc[0]
        vol_p = vol_row["Day-0 p-value"]
        vol_ci = vol_row[f"Day-0 CI ({ci_choice})"]

        if vol_p < 0.05:
            st.markdown(f"""
            **Volatility (Std Dev):**  
            • Earnings-day volatility is **significantly higher** than normal.  
            • CI: **{vol_ci}**  
            → Market uncertainty spikes around earnings, leading to wider price swings.
            """)
        else:
            st.markdown(f"""
            **Volatility (Std Dev):**  
            • No significant evidence of elevated volatility on earnings day.  
            • CI: **{vol_ci}**
            """)

        # ---------------------------
        # % POSITIVE DAYS INTERPRETATION
        # ---------------------------
        pos_row = res_df.loc[res_df["Statistic"] == "% Positive Days"].iloc[0]
        pos_p = pos_row["Day-0 p-value"]
        pos_ci = pos_row[f"Day-0 CI ({ci_choice})"]

        if pos_p < 0.05:
            st.markdown(f"""
            **% Positive Days:**  
            • Earnings-day returns are **more likely to be positive** than normal.  
            • CI: **{pos_ci}**
            """)
        else:
            st.markdown(f"""
            **% Positive Days:**  
            • No statistical evidence that earnings days have a directional bias.  
            • CI: **{pos_ci}**  
            → Direction of earnings reactions remains unpredictable.
            """)

        st.markdown("<hr>", unsafe_allow_html=True)

        # ---------------------------
        # SUMMARY
        # ---------------------------

        st.markdown(
            "<h3 style='font-size:18px; font-weight:650; margin-bottom:4px;'>"
            "🧾 Statistical Significance Summary"
            "</h3>",
            unsafe_allow_html=True
        )

        st.markdown("""
        **1️⃣ Earnings days create statistically meaningful effects**, especially in volatility.  
        - The market reacts strongly to new information, widening trading ranges.  
        - Idiosyncratic volatility also increases, confirming that the reaction is firm-specific.

        **2️⃣ Return direction is not predictable.**  
        - Positive surprises and negative surprises both occur,  
          and there is no consistent tendency toward positive reactions.

        **3️⃣ Large price reactions mainly occur in a subset of earnings events.**  
        - This explains why the mean may appear significant even when the median is not.

        **4️⃣ Post-earnings volatility can shift structurally.**  
        - Suggests the market reassesses NVIDIA’s risk profile after earnings.

""")

