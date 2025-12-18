# tabs/event_analysis.py

"""
Earnings Event Analysis Tab
Author : Nawaz Pasha
---------------------------

This module renders the **Earnings Event Analysis** tab of the dashboard.

Purpose:
- Analyze NVIDIA’s return behavior around earnings events
- Decompose returns into Total, Factor, and Idiosyncratic components
- Compute pre / event / post averages
- Evaluate interval-level earnings moves
- Visualize event-aligned return patterns
- Perform EWMA drift analysis
- Summarize earnings patterns and statistics
- Rank top positive / negative earnings windows
- Compare events side-by-side
- Perform clustering and similarity analysis on event reactions

Inputs:
- df      : Daily return dataframe
- df_e    : Earnings dates dataframe
- ev_out  : Output of build_event_window (mats, idxs, event_info)
- win     : Event window size (± days)

Outputs:
- Streamlit-rendered charts, KPIs, tables
- No data mutation (presentation layer only)

Design Notes:
- Heavy numerical computations rely on analysis.py
- Plot construction relies on plots.py
- UI layout and HTML are defined locally to preserve exact behavior
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from constants import COLORS
from analysis import (
    compute_event_averages,
    compute_pre_post_means,
    compute_interval_move,
    compute_progressive_curves,
    extract_pattern_table_values,
    summarize_pattern_table,
    compute_stat_table,
    compute_top_bottom,
    compute_risk_ratios,
    build_event_comparison_table,
)
from plots import (
    apply_plotly_layout,
    plot_avg_returns_selected,
)


def render_event_analysis(df, df_e, ev_out, win):
    """
    Renders the Earnings Event Analysis tab.

    Called from app.py inside the Tab-2 context.
    Assumes ev_out has already been constructed upstream.
    """

    # ---------------------------------------------------------------
    # PAGE HEADER
    # ---------------------------------------------------------------
    st.markdown(
        "<h2 style='font-size:22px; font-weight:700; margin-bottom:4px;'>"
        "📊 Earnings Event Analysis"
        "</h2>",
        unsafe_allow_html=True
    )

    if not ev_out:
        st.warning("No earnings windows fully inside the sample for the selected window length / date range.")
        st.stop()

    mats, idxs, event_info = ev_out

    # ---------------------------------------------------------------
    # BASIC EVENT-ALIGNED AGGREGATES
    # ---------------------------------------------------------------
    idxs2, avg = compute_event_averages(ev_out)

    (
        pre_tot, ev_tot, post_tot,
        pre_fac, ev_fac, post_fac,
        pre_idi, ev_idi, post_idi
    ) = compute_pre_post_means(avg, idxs2)

    # ---------------------------------------------------------------
    # KPI TILE FUNCTION (UNIFORM STYLING)
    # ---------------------------------------------------------------
    def kpi_tile(label, value, icon="📊", color="#4a4a4a", is_ratio=False):

        formatted = (
            f"{value:.2f}" if is_ratio else
            f"{value:.2%}" if isinstance(value, (int, float)) else value
        )

        if isinstance(value, (int, float)):
            col_val = "green" if value > 0 else "red" if value < 0 else "#000"
        else:
            col_val = "#000"

        tile_html = f"""
        <div style="
            border-radius:8px;
            padding:6px 8px;
            background-color:white;
            border:1px solid #e0e0e0;
            width:150px;
            height:85px;
            display:flex;
            flex-direction:column;
            justify-content:center;
            align-items:center;
            box-shadow:0 1px 2px rgba(0,0,0,0.05);
        ">
            <div style="font-size:16px; color:{color};">{icon}</div>
            <div style="font-size:11px; color:#666; font-weight:600; text-align:center; line-height:1.15;">
                {label}
            </div>
            <div style="font-size:15px; font-weight:700; color:{col_val};">
                {formatted}
            </div>
        </div>
        """
        return tile_html

    # =====================================================================
    # Pre / Event / Post KPIs
    # =====================================================================
    st.markdown("<h3 style='font-size:17px; margin-top:6px;'>📌 Earnings Event KPIs</h3>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:12px; line-height:1.3;'>Summarizes NVIDIA’s average behavior before, on, and after earnings, broken into total, factor, and idiosyncratic effects.</p>", unsafe_allow_html=True)

    
    st.markdown("<h4 style='font-size:14px; margin-bottom:4px;'>🔹 Pre / Event / Post KPIs</h4>",
                unsafe_allow_html=True)
    
    
    colA1, colA2, colA3 = st.columns(3)

    with colA1:
        st.markdown(kpi_tile("Pre Total", pre_tot, "📉"), unsafe_allow_html=True)
        st.markdown(kpi_tile("Pre Factor", pre_fac, "⚙️"), unsafe_allow_html=True)
        st.markdown(kpi_tile("Pre Idio", pre_idi, "🎯"), unsafe_allow_html=True)

    with colA2:
        st.markdown(kpi_tile("Event Total (Day 0)", ev_tot, "⭐"), unsafe_allow_html=True)
        st.markdown(kpi_tile("Event Factor", ev_fac, "📊"), unsafe_allow_html=True)
        st.markdown(kpi_tile("Event Idio", ev_idi, "💥"), unsafe_allow_html=True)

    with colA3:
        st.markdown(kpi_tile("Post Total", post_tot, "📈"), unsafe_allow_html=True)
        st.markdown(kpi_tile("Post Factor", post_fac, "⚡"), unsafe_allow_html=True)
        st.markdown(kpi_tile("Post Idio", post_idi, "🔧"), unsafe_allow_html=True)

    st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)

    # =====================================================================
    # Interval Move KPI Summary
    # =====================================================================
    st.markdown("<h3 style='font-size:17px;'>🔹 KPI Summary (Event-Wise Interval Moves)</h3>",
                unsafe_allow_html=True)

    interval_df = compute_interval_move(df, df_e, win)
    interval_returns = interval_df["IntervalReturn"].dropna()

    pos_pct = (interval_returns > 0).mean()
    neg_pct = (interval_returns < 0).mean()
    ratio_pos_neg = pos_pct / neg_pct if neg_pct else np.nan

    avg_move = interval_returns.mean()
    med_move = interval_returns.median()
    ratio_med_avg = med_move / avg_move if avg_move else np.nan

    avg_up = interval_returns[interval_returns > 0].mean()
    avg_down = interval_returns[interval_returns < 0].mean()
    ratio_up_down = abs(avg_up) / abs(avg_down) if avg_down else np.nan

    max_up = interval_returns.max()
    max_down = interval_returns.min()
    ratio_max = abs(max_up) / abs(max_down) if max_down else np.nan
    
    st.markdown("<p style='font-size:12px; line-height:1.3;'>Shows how often earnings windows were positive or negative and how large typical up-moves and down-moves were.</p>", unsafe_allow_html=True)


    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.markdown(kpi_tile("Positive %", pos_pct, "📈"), unsafe_allow_html=True)
        st.markdown(kpi_tile("Negative %", neg_pct, "📉"), unsafe_allow_html=True)
        st.markdown(kpi_tile("Pos/Neg Ratio", ratio_pos_neg, "⚖️", is_ratio=True), unsafe_allow_html=True)

    with k2:
        st.markdown(kpi_tile("Avg Move", avg_move, "📊"), unsafe_allow_html=True)
        st.markdown(kpi_tile("Median Move", med_move, "📐"), unsafe_allow_html=True)
        st.markdown(kpi_tile("Median/Avg", ratio_med_avg, "📏", is_ratio=True), unsafe_allow_html=True)

    with k3:
        st.markdown(kpi_tile("Avg Up", avg_up, "🟢"), unsafe_allow_html=True)
        st.markdown(kpi_tile("Avg Down", avg_down, "🔻"), unsafe_allow_html=True)
        st.markdown(kpi_tile("|Up|/|Down|", ratio_up_down, "💹", is_ratio=True), unsafe_allow_html=True)

    with k4:
        st.markdown(kpi_tile("Max Up", max_up, "🚀"), unsafe_allow_html=True)
        st.markdown(kpi_tile("Max Down", max_down, "📉"), unsafe_allow_html=True)
        st.markdown(kpi_tile("Up/Down Max Ratio", ratio_max, "📊", is_ratio=True), unsafe_allow_html=True)

    st.markdown("<div style='margin-top:14px;'></div>", unsafe_allow_html=True)

    # =====================================================================
    # EARNINGS REACTION VISUALS
    # =====================================================================
    
    st.markdown("<h3 style='font-size:17px; margin-top:2px;'>📈 Earnings Reaction Visuals</h3>",
                unsafe_allow_html=True)
    
    with st.expander("📊 Return Distribution & Market Reaction", expanded=True):
    
        # ------------------------------------------------------------
        # 2 CHARTS SIDE BY SIDE
        # ------------------------------------------------------------
        left_col, right_col = st.columns(2)
    
        # =================================================================
        # LEFT CHART — Combined:
        #   • Average Daily Returns Around Earnings
        #   • Progressive Average Curve
        #   • Progressive Median Curve
        # =================================================================
        with left_col:
            
    
            st.markdown(
                "<h4 style='font-size:13px; margin-bottom:0px;'>"
                "📉 Avg Daily Returns + Progressive Avg/Median"
                "</h4>",
                unsafe_allow_html=True
            )
            
            st.markdown("<p style='font-size:12px; line-height:1.3;'>Displays the typical return pattern around earnings along with rolling average and median curves, revealing consistent directional trends.</p>", unsafe_allow_html=True)

    
            # Fetch curves
            fig_comb = go.Figure()
    
            # 1) Average Daily Curve (Total, Systematic, Idio)
            fig_comb2 = plot_avg_returns_selected(
                idxs2, avg, ["Total", "Systematic", "Idiosyncratic"]
            )
    
            # Transfer traces into our combined figure
            for tr in fig_comb2.data:
                fig_comb.add_trace(tr)
    
            # 2) Progressive Curves
            curves = compute_progressive_curves(mats, idxs)
    
            fig_comb.add_trace(go.Scatter(
                x=curves["idxs"], y=curves["avg"],
                name="Progressive Avg",
                line=dict(width=2.5, dash="solid")
            ))
    
            fig_comb.add_trace(go.Scatter(
                x=curves["idxs"], y=curves["median"],
                name="Progressive Median",
                line=dict(width=2.2, dash="dot")
            ))
    
            # Event-day vertical marker
            fig_comb.add_vline(x=0, line_color="red", line_dash="dash")
    
            fig_comb = apply_plotly_layout(fig_comb, title="", ytitle="Return")
            fig_comb.update_layout(height=320)
    
            st.plotly_chart(fig_comb, use_container_width=True)
    
        # =================================================================
        # RIGHT CHART — Boxplot (Earnings vs Non-Earnings)
        # =================================================================
        with right_col:
    
            st.markdown(
                "<h4 style='font-size:13px; margin-bottom:0px;'>"
                "📦 Boxplot: Earnings Day vs Non-Earnings Days"
                "</h4>",
                unsafe_allow_html=True
            )
            
            st.markdown("<p style='font-size:12px; line-height:1.3;'>Compares NVIDIA’s return distribution on earnings days versus normal days to show differences in volatility and typical movement.</p>", unsafe_allow_html=True)

    
            # Identify earnings days
            event_day_dates = pd.to_datetime(event_info["event_date"].dropna().unique())
            mask = df["Date"].isin(event_day_dates)
    
            earn_r = df.loc[mask, "NVDA"].dropna()
            non_r = df.loc[~mask, "NVDA"].dropna()
    
            fig_box = go.Figure()
            fig_box.add_trace(go.Box(
                y=earn_r,
                name="Earnings Day",
                boxmean=True,
                marker_color=COLORS["blue"]
            ))
            fig_box.add_trace(go.Box(
                y=non_r,
                name="Non-Earnings Days",
                boxmean=True,
                marker_color=COLORS["gray"]
            ))
    
            fig_box = apply_plotly_layout(fig_box, title="", ytitle="Return")
            fig_box.update_layout(height=320)
    
            st.plotly_chart(fig_box, use_container_width=True)
    
    # Small spacing buffer
    st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)

    # =====================================================================
    # TABLES, STATISTICS, EWMA, TOP RETURNS, COMPARISON TABLE
    # =====================================================================
    
    # =====================================================================
    # EWMA RETURN ANALYTICS
    # =====================================================================
    
    st.markdown("<h3 style='font-size:17px; margin-top:10px;'>📉 EWMA Return Behavior</h3>", 
                unsafe_allow_html=True)
    
    with st.expander("📘 EWMA-Smoothed Curve, Abnormal Return Index, KPIs & Drift Classification", expanded=True):
    
        # ------------------------------------------------------------
        # USER CONTROL: Lambda Slider
        # ------------------------------------------------------------
        st.markdown("<h4 style='font-size:14px;'>⚙️ EWMA Settings</h4>", unsafe_allow_html=True)
    
        lambda_ewma = st.slider(
            "EWMA Smoothing λ (Higher = smoother)",
            min_value=0.05,
            max_value=0.95,
            value=0.40,
            step=0.05
        )
    
        # Always use TOTAL return curve
        raw_curve = avg["total"]
        days = np.arange(-win, win + 1)
    
        # ------------------------------------------------------------
        # COMPUTE EWMA-SMOOTHED AVERAGE RETURN CURVE
        # ------------------------------------------------------------
        ewma_curve = np.zeros_like(raw_curve)
        ewma_curve[0] = raw_curve[0]
        for i in range(1, len(raw_curve)):
            ewma_curve[i] = lambda_ewma * ewma_curve[i-1] + (1 - lambda_ewma) * raw_curve[i]
    
        # ------------------------------------------------------------
        # COMPUTE DRIFT KPIs
        # ------------------------------------------------------------
        pre_slope = np.mean(np.diff(ewma_curve[:win])) if win > 1 else 0
        post_slope = (
            np.mean(np.diff(ewma_curve[win+1:])) 
            if win+1 < len(ewma_curve) - 1 else 0
        )
        drift_ratio = (post_slope / pre_slope) if pre_slope != 0 else np.nan
    
        # ------------------------------------------------------------
        # DRIFT CLASSIFICATION BADGE
        # ------------------------------------------------------------
        def classify(pre, post):
            if abs(pre) < 0.00005 and abs(post) < 0.00005:
                return "🟡 Flat / No Drift"
            if pre > 0 and post > 0:
                return "🟢 Bullish Drift"
            if pre < 0 and post < 0:
                return "🔴 Bearish Drift"
            if pre > 0 and post < 0:
                return "🟣 Post-Earnings Reversal (Bull → Bear)"
            if pre < 0 and post > 0:
                return "🔵 Post-Earnings Acceleration (Bear → Bull)"
            return "⚪ Mixed Drift Pattern"
    
        drift_label = classify(pre_slope, post_slope)
    
        # ------------------------------------------------------------
        # KPI TILE FORMAT
        # ------------------------------------------------------------
        def kpi_tile_small(label, value, icon="📊"):
            if isinstance(value, (float, int)):
                col = "green" if value > 0 else "red" if value < 0 else "#000"
                formatted = f"{value:.4f}"
            else:
                col = "#000"
                formatted = value
    
            return f"""
            <div style="
                border-radius:8px;
                padding:5px 6px;
                background-color:white;
                border:1px solid #e0e0e0;
                width:150px;
                height:70px;
                display:flex;
                flex-direction:column;
                justify-content:center;
                align-items:center;
                box-shadow:0 1px 2px rgba(0,0,0,0.05);
            ">
                <div style="font-size:15px;">{icon}</div>
                <div style="font-size:11px; color:#666; font-weight:600; text-align:center;">{label}</div>
                <div style="font-size:14px; color:{col}; font-weight:700;">{formatted}</div>
            </div>
            """
    
        # ------------------------------------------------------------
        # EWMA KPIs
        # ------------------------------------------------------------
        st.markdown("<h4 style='font-size:14px; margin-bottom:6px;'>🔹 EWMA Drift KPIs</h4>", 
                    unsafe_allow_html=True)
    
        k1, k2, k3, k4 = st.columns(4)
    
        with k1:
            st.markdown(kpi_tile_small("Pre-Earnings Drift", pre_slope, "⬅️"),
                        unsafe_allow_html=True)
        with k2:
            st.markdown(kpi_tile_small("Post-Earnings Drift", post_slope, "➡️"),
                        unsafe_allow_html=True)
        with k3:
            st.markdown(kpi_tile_small("Drift Ratio (Post/Pre)", drift_ratio, "⚖️"),
                        unsafe_allow_html=True)
        with k4:
            st.markdown(
                f"""
                <div style="
                    border-radius:8px;
                    padding:5px 6px;
                    background-color:white;
                    border:1px solid #e0e0e0;
                    width:150px;
                    height:70px;
                    display:flex;
                    flex-direction:column;
                    justify-content:center;
                    align-items:center;
                    box-shadow:0 1px 2px rgba(0,0,0,0.05);
                    font-size:12px;
                    font-weight:700;
                    color:#333;
                ">{drift_label}</div>
                """,
                unsafe_allow_html=True
            )
    
        # ------------------------------------------------------------
        # SIDE-BY-SIDE CHARTS
        # ------------------------------------------------------------
        left_col, right_col = st.columns(2)
    
        # LEFT — EWMA-smoothed return curve
        with left_col:
            st.markdown(
                "<h4 style='font-size:13px; margin-bottom:0px;'>📈 EWMA-Smoothed Total Return Curve</h4>",
                unsafe_allow_html=True
            )
            
            st.markdown("<p style='font-size:12px; line-height:1.3;'>Smooths the average event-aligned return curve to highlight drift patterns before and after earnings.</p>", unsafe_allow_html=True)

    
            fig_ewma = go.Figure()
            fig_ewma.add_trace(go.Scatter(
                x=days,
                y=raw_curve,
                name="Raw Curve",
                opacity=0.45,
                line=dict(width=1.2, color="gray")
            ))
            fig_ewma.add_trace(go.Scatter(
                x=days,
                y=ewma_curve,
                name=f"EWMA Smoothed (λ={lambda_ewma})",
                line=dict(width=3)
            ))
            fig_ewma.add_vline(x=0, line_color="red", line_dash="dash")
    
            fig_ewma = apply_plotly_layout(fig_ewma, title="", ytitle="Return")
            fig_ewma.update_layout(height=270)
            st.plotly_chart(fig_ewma, use_container_width=True)
    
        # RIGHT — Abnormal return index
        with right_col:
            st.markdown(
                "<h4 style='font-size:13px; margin-bottom:0px;'>📊 EWMA Abnormal Return Index</h4>",
                unsafe_allow_html=True
            )
            
            st.markdown("<p style='font-size:12px; line-height:1.3;'>Measures how unusual each earnings-day move was relative to NVIDIA’s normal trading behavior.</p>", unsafe_allow_html=True)

    
            baseline = df.loc[~df["Date"].isin(df_e["EarningsDate"]), "NVDA"].median()
            event_day_returns = mats["NVDA"][:, win]
            abnormal = event_day_returns - baseline
    
            lambda_cycle = 0.30
            ewma_cycle = np.zeros_like(abnormal)
            ewma_cycle[0] = abnormal[0]
    
            for i in range(1, len(abnormal)):
                ewma_cycle[i] = lambda_cycle * ewma_cycle[i-1] + (1 - lambda_cycle) * abnormal[i]
    
            fig_cycle = go.Figure()
            fig_cycle.add_trace(go.Scatter(
                x=np.arange(1, len(abnormal) + 1),
                y=abnormal,
                name="Abnormal Return",
                mode="markers+lines",
                marker=dict(size=6),
                line=dict(width=1.2)
            ))
            fig_cycle.add_trace(go.Scatter(
                x=np.arange(1, len(ewma_cycle) + 1),
                y=ewma_cycle,
                name=f"EWMA Cycle (λ={lambda_cycle})",
                mode="lines",
                line=dict(width=3)
            ))
    
            fig_cycle = apply_plotly_layout(fig_cycle, title="", ytitle="Abnormal Return")
            fig_cycle.update_layout(height=270)
            st.plotly_chart(fig_cycle, use_container_width=True)
    
    st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)
    
    
    # =====================================================================
    # EARNINGS PATTERN TABLE + SUMMARY
    # =====================================================================
    
    st.markdown("<h3 style='font-size:17px; margin-top:4px;'>📘 Earnings Pattern Table + Summary</h3>",
                unsafe_allow_html=True)
    st.markdown("<p style='font-size:12px; line-height:1.3;'>Summarizes per-event return characteristics and aggregates common patterns across all earnings events.The table has summary statistics scroll down the same table to view the same</p>", unsafe_allow_html=True)

    
    with st.expander("📘 Expanded Earnings Pattern Table & Summary", expanded=True):
    
        pat = extract_pattern_table_values(df, df_e, win).replace({None: ""})

        pat2 = pat.copy()
        pat2.insert(0, "Metric", pat2["EarningsDate"].dt.strftime("%Y-%m-%d"))
        pat2 = pat2.drop(columns=["EarningsDate"])


    
        summary = summarize_pattern_table(pat)
    
        def prep_block(block):
            b = block.copy().replace({None: ""})
            if "Metric" not in b.columns:
                b.insert(0, "Metric", b.index)
            else:
                b = b[["Metric"] + [c for c in b.columns if c != "Metric"]]
            return b
    
        A = prep_block(summary["A"])
        B = prep_block(summary["B"])
        C = prep_block(summary["C"])
    
        def title_row(text, cols):
            return pd.DataFrame([[text] + [""] * (len(cols) - 1)], columns=cols)
    
        stacked = pd.concat([
            pat2,
            title_row("Simple Average Returns – Last 12 Earnings", A.columns), A,
            title_row("Absolute Average Returns – Last 12 Earnings", B.columns), B,
            title_row("Median, Minimum, Maximum Returns – Last 12 Earnings", C.columns), C,
        ], ignore_index=True)
        
        
        stacked = stacked.copy()
        
        # Rename CustomInterval → InternalRange Values
        if "CustomInterval" in stacked.columns:
            stacked = stacked.rename(columns={"CustomInterval": "InternalRange Values"})
        
        # Ensure IntervalRange has no 'None'
        if "IntervalRange" in stacked.columns:
            stacked["IntervalRange"] = stacked["IntervalRange"].fillna("")
        
        # Desired front column order
        front_cols = ["Metric"]
        
        if "IntervalRange" in stacked.columns:
            front_cols.append("IntervalRange")
        
        if "InternalRange Values" in stacked.columns:
            front_cols.append("InternalRange Values")
        
        # Preserve order of remaining columns
        remaining_cols = [c for c in stacked.columns if c not in front_cols]
        
        # Reorder final dataframe
        stacked = stacked[front_cols + remaining_cols]

    
        def style_stacked(df):
            title_bg = {
                "Simple Average": "#d9e6ff",
                "Absolute Average": "#ffe6cc",
                "Median, Minimum": "#fff3cd"
            }
    
            def highlight_titles(row):
                t = str(row["Metric"])
                for key, bg in title_bg.items():
                    if key in t:
                        return [f"background:{bg}; font-weight:bold; text-align:center;"] * len(row)
                return [""] * len(row)
    
            def val_color(v):
                if isinstance(v, float):
                    if v > 0:
                        return "color:green;" #font-weight:700;"
                    elif v < 0:
                        return "color:red;" #font-weight:700;"
                return ""
    
            return (
                df.style
                .apply(highlight_titles, axis=1)
                .applymap(val_color)
                .format(lambda x: f"{x:.2%}" if isinstance(x, float) else x)
                .set_table_styles([
                    {"selector": "th",
                     "props": "font-weight:bold; background:#e9ecef; text-align:center; font-size:12px;"}
                ])
                .set_properties(subset=["Metric"], **{"font-weight": "bold"})
                .set_properties(**{"text-align": "center", "border": "1px solid #ccc", "font-size": "12px"})
            )
    
        #st.dataframe(style_stacked(stacked), use_container_width=True, hide_index=True)
        st.dataframe(
        style_stacked(stacked),
        use_container_width=True,
        hide_index=True,
        height=700
    )

    st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)
    
    
    # =====================================================================
    # STATISTICAL SUMMARY + RISK RATIOS
    # =====================================================================
    
    st.markdown("<h3 style='font-size:17px; margin-top:4px;'>📐 Statistical Summary & ⚖️ Risk Ratios</h3>",
                unsafe_allow_html=True)
    
    st.markdown("<p style='font-size:12px; line-height:1.3;'>Provides distribution metrics and risk-adjusted return ratios to evaluate the quality of earnings-related returns.</p>", unsafe_allow_html=True)

    
    with st.expander("📐 Expanded Statistical Summary & Risk Ratios", expanded=True):
    
        colA, colB = st.columns(2)
    
        # -------- Statistical Summary --------
        stat_table = compute_stat_table(interval_returns)
        stat_table_fmt = stat_table.applymap(
            lambda x: f"{x:.2%}" if isinstance(x, (int, float)) else x
        )
    
        def color_stat(v):
            try:
                if isinstance(v, str) and "%" in v:
                    cleaned = v.replace("%", "").strip()
                    num = float(cleaned) / 100
                    if num > 0:
                        return "color:green;"# font-weight:700;"
                    elif num < 0:
                        return "color:red;" #font-weight:700;"
            except:
                pass
            return ""
    
        with colA:
            st.markdown("<h4 style='font-size:14px;'>📐 Statistical Summary</h4>",
                        unsafe_allow_html=True)
            st.dataframe(
                stat_table_fmt.style
                .applymap(color_stat)
                .set_table_styles([
                    {"selector": "th",
                     "props": "font-weight:bold; background:#e9ecef; text-align:center;"}
                ])
                .set_properties(**{"text-align": "center", "font-size": "12px"}),
                use_container_width=True,
                hide_index=True
            )
    
        # -------- Risk Ratios --------
        rr = compute_risk_ratios(interval_returns).applymap(
            lambda x: f"{x:.4f}" if isinstance(x, (float, int)) else x
        )
    
        with colB:
            st.markdown("<h4 style='font-size:14px;'>⚖️ Return / Risk Ratios</h4>",
                        unsafe_allow_html=True)
            st.dataframe(
                rr.style
                .set_table_styles([
                    {"selector": "th",
                     "props": "font-weight:bold; background:#e9ecef; text-align:center;"}
                ])
                .set_properties(**{"text-align": "center", "font-size": "12px"}),
                use_container_width=True,
                hide_index=True
            )
    
    st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)
    
    
    # =====================================================================
    # TOP POSITIVE / NEGATIVE RETURNS
    # =====================================================================
    
    st.markdown("<h3 style='font-size:17px; margin-top:4px;'>📉 Top Interval Returns</h3>",
                unsafe_allow_html=True)
    st.markdown("<p style='font-size:12px; line-height:1.3;'>Highlights NVIDIA’s biggest positive and negative earnings-period return events.</p>", unsafe_allow_html=True)

    
    with st.expander("📉 Top Positive & Negative Interval Returns", expanded=True):
    
        top_pos, top_neg = compute_top_bottom(interval_df, n=5)
    
        df_pos = top_pos[["Range", "IntervalReturn"]].reset_index(drop=True)
        df_neg = top_neg[["Range", "IntervalReturn"]].reset_index(drop=True)
    
        def style_top(df, positive=True):
            title = f"Top {len(df)} {'Positive' if positive else 'Negative'}"
            title_color = "#d6f5d6" if positive else "#ffd6d6"
    
            header = pd.DataFrame([[title, ""]], columns=df.columns)
            df2 = pd.concat([header, df], ignore_index=True)
    
            def header_highlight(row):
                return (
                    [f"background:{title_color}; font-weight:bold;"]
                    * len(row) if row.name == 0 else [""] * len(row)
                )
    
            def val_color(v):
                if isinstance(v, float):
                    if v > 0:
                        return "color:green;"# font-weight:700;"
                    elif v < 0:
                        return "color:red;"# font-weight:700;"
                return ""
    
            return (
                df2.style
                .apply(header_highlight, axis=1)
                .applymap(val_color, subset=["IntervalReturn"])
                .set_table_styles([
                    {"selector": "th",
                     "props": "font-weight:bold; background:#e9ecef; text-align:center;"}
                ])
                .set_properties(subset=["Range"], **{"font-weight": "bold"})
                .set_properties(**{"text-align": "center", "font-size": "12px"})
                .format(lambda x: f"{x:.2%}" if isinstance(x, float) else x)
            )
    
        c1, c2 = st.columns(2)
    
        with c1:
            st.markdown("<h4 style='font-size:14px;'>📈 Top Positive</h4>",
                        unsafe_allow_html=True)
            st.dataframe(style_top(df_pos, positive=True),
                         use_container_width=True, hide_index=True)
    
        with c2:
            st.markdown("<h4 style='font-size:14px;'>📉 Top Negative</h4>",
                        unsafe_allow_html=True)
            st.dataframe(style_top(df_neg, positive=False),
                         use_container_width=True, hide_index=True)
    
    st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)
    
    
    # =====================================================================
    # EARNINGS EVENT COMPARISON TABLE
    # =====================================================================
    
    st.markdown("<h3 style='font-size:17px; margin-top:4px;'>🧾 Earnings Event Comparison Table</h3>",
                unsafe_allow_html=True)
    
    st.markdown("<p style='font-size:12px; line-height:1.3;'>Compares each earnings event side-by-side to show which quarters experienced the strongest or weakest reactions.</p>", unsafe_allow_html=True)

    
    with st.expander("🧾 Earnings Event Comparison Table", expanded=True):
    
        comparison_df = build_event_comparison_table(df, df_e, win)
    
        formatted_df = comparison_df.applymap(
            lambda v: f"{v:.2%}" if isinstance(v, (float, int)) else v
        )
    
        def zebra(df):
            z = pd.DataFrame("", index=df.index, columns=df.columns)
            for i in range(len(df)):
                z.iloc[i, :] = "background-color:white;" if i % 2 == 0 else "background-color:#fafafa;"
            return z
    
        def first_bold(df): 
            s = pd.DataFrame("", index=df.index, columns=df.columns) 
            s.iloc[:, 0] = "font-weight:bold; background:#f2f4f7;" 
            return s

    
        def color_values(v):
            if isinstance(v, str) and v.endswith("%"):
                num = float(v.replace("%", "")) / 100
                if num > 0:
                    return "color:green;"# font-weight:700;"
                if num < 0:
                    return "color:red;"# font-weight:700;"
            return ""
    
        styled_cmp = (
            formatted_df.style
            # .apply(first_bold, axis=None)
            .apply(zebra, axis=None)
            .applymap(color_values)
            .set_table_styles([
                {"selector": "th",
                 "props": "font-weight:bold; background:#e9ecef; text-align:center; font-size:12px;"},
                {"selector": "td",
                 "props": "padding:6px; border:1px solid #ddd; font-size:12px;"}
            ])
            .set_properties(**{"text-align": "center"})
        )
    
        st.dataframe(styled_cmp, use_container_width=True, hide_index=False)
        
        # =====================================================================
        # ADVANCED INSIGHTS — EVENT CLUSTERING & SIMILARITY ENGINE
        # =====================================================================
        
        st.markdown("<h3 style='font-size:17px; margin-top:14px;'>🧠 Advanced Earnings Insights</h3>",
                    unsafe_allow_html=True)
        
        st.markdown("<p style='font-size:12px; line-height:1.3;'>Groups earnings events into clusters based on how the return curves behaved, revealing recurring reaction types.</p>", unsafe_allow_html=True)

        
        with st.expander("📘 Event Reaction Clustering & Similarity Analysis", expanded=True):
        
            # ------------------------------------------------------------
            # 1️⃣ PREP DATA
            # ------------------------------------------------------------
            X = mats["NVDA"]  
            days = idxs2
            num_events = X.shape[0]
        
            # ------------------------------------------------------------
            # 2️⃣ K-MEANS CLUSTERING
            # ------------------------------------------------------------
            from sklearn.preprocessing import StandardScaler
            from sklearn.cluster import KMeans
        
            k = 3
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X)
        
            km = KMeans(n_clusters=k, n_init=20, random_state=1)
            labels = km.fit_predict(Xs)
        
            cluster_curves = {c: X[labels == c].mean(axis=0) for c in range(k)}
        
            # ------------------------------------------------------------
            # 3️⃣ CLUSTER KPI TILES
            # ------------------------------------------------------------
            def kpi_cluster(label, val, icon):
                return f"""
                <div style="
                    width:110px;
                    height:62px;
                    border-radius:6px;
                    padding:4px;
                    border:1px solid #ddd;
                    background:white;
                    display:flex;
                    flex-direction:column;
                    align-items:center;
                    justify-content:center;
                    box-shadow:0 1px 2px rgba(0,0,0,0.04);
                ">
                    <div style='font-size:14px;'>{icon}</div>
                    <div style='font-size:10px;color:#666;font-weight:600;text-align:center;'>
                        {label}
                    </div>
                    <div style='font-size:13px;font-weight:700;'>
                        {val}
                    </div>
                </div>
                """
        
            st.markdown("<h4 style='font-size:14px;'>🔹 Cluster Summary</h4>", unsafe_allow_html=True)
        
            colA, colB, colC = st.columns(3)
        
            with colA:
                st.markdown(kpi_cluster("Cluster 0 Size", np.sum(labels == 0), "🟦"), unsafe_allow_html=True)
                st.markdown(kpi_cluster("Event-0 Avg", f"{cluster_curves[0][win]:.3%}", "⭐"), unsafe_allow_html=True)
        
            with colB:
                st.markdown(kpi_cluster("Cluster 1 Size", np.sum(labels == 1), "🟩"), unsafe_allow_html=True)
                st.markdown(kpi_cluster("Event-0 Avg", f"{cluster_curves[1][win]:.3%}", "⭐"), unsafe_allow_html=True)
        
            with colC:
                st.markdown(kpi_cluster("Cluster 2 Size", np.sum(labels == 2), "🟥"), unsafe_allow_html=True)
                st.markdown(kpi_cluster("Event-0 Avg", f"{cluster_curves[2][win]:.3%}", "⭐"), unsafe_allow_html=True)
        
            # ------------------------------------------------------------
            # CLUSTER CURVE PLOT
            # ------------------------------------------------------------
            st.markdown("<h4 style='font-size:14px; margin-top:10px;'>📈 Cluster-Averaged Return Curves</h4>",
                        unsafe_allow_html=True)
        
            fig_cl = go.Figure()
            colors_c = {0: "#4e79a7", 1: "#59a14f", 2: "#e15759"}
        
            for c in range(k):
                fig_cl.add_trace(go.Scatter(
                    x=days,
                    y=cluster_curves[c],
                    name=f"Cluster {c}",
                    line=dict(width=2.5, color=colors_c[c])
                ))
        
            fig_cl.add_vline(x=0, line_color="black", line_dash="dash")
            fig_cl = apply_plotly_layout(fig_cl, "", "Return")
            fig_cl.update_layout(height=260)
        
            st.plotly_chart(fig_cl, use_container_width=True, key="cluster_curve_plot")
        
            # =================================================================
            # 5️⃣ SIMILARITY ENGINE — TOP-3 CLOSEST EVENTS
            # =================================================================
            st.markdown("<h4 style='font-size:14px; margin-top:14px;'>🔎 Event Similarity Engine (Top-3 Nearest)</h4>",
                        unsafe_allow_html=True)
            
            st.markdown("<p style='font-size:12px; line-height:1.3;'>Finds the most similar historical earnings reactions for each event based on curve resemblance.</p>", unsafe_allow_html=True)

        
            from sklearn.metrics.pairwise import cosine_distances
            dist = cosine_distances(Xs)
        
            event_dates = event_info["event_date"].dt.strftime("%Y-%m-%d").tolist()
        
            rows = []
            for i in range(num_events):
                dists = dist[i]
                nearest = np.argsort(dists)[1:4]
                rows.append({
                    "Event": event_dates[i],
                    "Nearest 1": event_dates[nearest[0]],
                    "Nearest 2": event_dates[nearest[1]],
                    "Nearest 3": event_dates[nearest[2]],
                })
        
            sim_df = pd.DataFrame(rows)
        
            # ------------------------------------------------------------
            # NEAREST-NEIGHBOR TABLE (compact)
            # ------------------------------------------------------------
            st.markdown("<h4 style='font-size:13px; margin-top:8px;'>📋 Nearest-Neighbor Table</h4>",
                        unsafe_allow_html=True)
        
            st.dataframe(
                sim_df.style.set_table_styles([
                    {"selector": "th", "props": "font-weight:bold; background:#e9ecef; text-align:center;"},
                    {"selector": "td", "props": "text-align:center; font-size:12px; border:1px solid #ddd;"}
                ]),
                use_container_width=True,
                hide_index=True
            )
        
            # ============================================================
            # END MODULE
            # ============================================================

