# tabs/overview.py
# tabs/overview.py

"""
Overview Tab Module
Author : Nawaz Pasha
-------------------

This module renders the **Overview** tab of the NVIDIA Earnings Analysis dashboard.

Purpose:
- Provide a high-level, executive summary of the dataset and analysis
- Explain the methodology used for return decomposition
- Present key descriptive statistics for Total, Systematic, and Idiosyncratic returns
- Visualize daily and cumulative return decomposition
- Highlight factor-model performance metrics
- Identify the most influential factors
- Show correlation structure among selected factors

Inputs:
- df            : Main daily return dataframe (filtered by date range)
- df_e          : Earnings dates dataframe (filtered to the same date range)
- selected_factors : List of factor names selected by the user
- win           : Event window size (± days around earnings)

Outputs:
- Streamlit-rendered UI components (KPIs, tables, charts)
- No data mutation; purely a presentation / visualization layer

Design Notes:
- This module contains **only UI logic** for the Overview tab
- All statistical computations are delegated to analysis.py
- All plotting helpers are delegated to plots.py
- Styling is intentionally embedded to preserve exact visual behavior
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go

from constants import COLORS
from analysis import get_model_decomposition_stats
from plots import plot_multitimeseries_with_event_lines, apply_plotly_layout


def render_overview(df, df_e, selected_factors, win):
    """
    Renders the Overview tab.

    This function is called from app.py inside the Overview tab context.
    It assumes that all input data has already been loaded, filtered,
    and prepared upstream.

    The function does not return anything — it directly renders
    Streamlit UI elements.
    """

    st.markdown(
        "<h2 style='font-size:22px; font-weight:700;'>📊 Overall NVIDIA Analysis Overview</h2>",
        unsafe_allow_html=True
    )


    # ---------------------------------------------------------------
    # 0️⃣ PROFESSIONAL COMPACT KPI TILE COMPONENT
    # ---------------------------------------------------------------
    def kpi_tile(label, value, icon="📊", color="#4a4a4a"):
        formatted = (
            f"{value:,}" if isinstance(value, int) else
            f"{value:.2%}" if isinstance(value, float) else value
        )

        tile_html = f"""
        <div style="
            border-radius:8px;
            padding:6px 8px;
            background-color:white;
            border:1px solid #e0e0e0;
            width:130px;
            height:78px;
            display:flex;
            flex-direction:column;
            justify-content:center;
            align-items:center;
            box-shadow:0 1px 2px rgba(0,0,0,0.05);
        ">
            <div style="font-size:16px; color:{color};">{icon}</div>
            <div style="font-size:10px; color:#666; font-weight:600; text-align:center;">{label}</div>
            <div style="font-size:14px; font-weight:700; color:#000;">
                {formatted}
            </div>
        </div>
        """
        return tile_html


    # ---------------------------------------------------------------
    # 1️⃣ UNIFIED DATASET OVERVIEW SECTION
    # ---------------------------------------------------------------
    st.markdown("<h3 style='font-size:17px;'>📌 Dataset Overview</h3>", unsafe_allow_html=True)

    in_range = df_e[(df_e["EarningsDate"] >= df["Date"].min()) &
                    (df_e["EarningsDate"] <= df["Date"].max())]

    nulls = df.isna().sum().sum()
    dups = df.duplicated(subset=["Date"]).sum()

    z_nvda = np.abs(stats.zscore(df["NVDA"], nan_policy="omit"))
    z_idio = np.abs(stats.zscore(df["Idio_Return"], nan_policy="omit"))
    out_nvda = int((z_nvda > 4).sum())
    out_idio = int((z_idio > 4).sum())

    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)

    with c1: st.markdown(kpi_tile("Days", df.shape[0], "🗓️"), unsafe_allow_html=True)
    with c2: st.markdown(kpi_tile("Factors", len(selected_factors), "🧠"), unsafe_allow_html=True)
    with c3: st.markdown(kpi_tile("Earnings", in_range.shape[0], "📆"), unsafe_allow_html=True)
    with c4: st.markdown(kpi_tile("Missing", int(nulls), "🧹"), unsafe_allow_html=True)
    with c5: st.markdown(kpi_tile("Duplicates", int(dups), "🔎"), unsafe_allow_html=True)
    with c6: st.markdown(kpi_tile("NVDA Outliers", out_nvda, "⚡"), unsafe_allow_html=True)
    with c7: st.markdown(kpi_tile("Idio Outliers", out_idio, "❗"), unsafe_allow_html=True)

    st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)


    # ---------------------------------------------------------------
    # 2️⃣ REDUCED-FONT METHODOLOGY
    # ---------------------------------------------------------------
    st.markdown("<h3 style='font-size:17px;'>Methodology</h3>", unsafe_allow_html=True)

    st.markdown(
        f"""
        <div style='font-size:12px; line-height:1.35;'>
        <b>How we do it:</b><br>
        1. Split each daily NVIDIA return into:<br>
        &nbsp;&nbsp;• <b>Systematic (Factor-based)</b> = betas × factor returns<br>
        &nbsp;&nbsp;• <b>Idiosyncratic</b> = NVIDIA minus factor-based<br>
        2. NVIDIA reports <b>after close</b> → <b>day 0 = next trading day</b>.<br>
        3. Use a <b>±{win}-day window</b> for each earnings event.<br>
        4. Compare <b>Total, Systematic, Idiosyncratic</b> returns & volatility shifts.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)


    # ---------------------------------------------------------------
    # 3️⃣ COMBINED SUMMARY STATISTICS TABLE
    # ---------------------------------------------------------------
    st.markdown("<h3 style='font-size:17px;'>📘 Combined Summary Statistics & Return Decomposition</h3>",
                unsafe_allow_html=True)

    series_map = {
        "Series": ["NVIDIA", "Systematic (Factors)", "Idiosyncratic"],
        "Count": [int(df["NVDA"].count()), int(df["Factor_Pred_Return"].count()), int(df["Idio_Return"].count())],
        "Mean": [df["NVDA"].mean(), df["Factor_Pred_Return"].mean(), df["Idio_Return"].mean()],
        "Volatility (Std Dev)": [df["NVDA"].std(), df["Factor_Pred_Return"].std(), df["Idio_Return"].std()],
        "Min": [df["NVDA"].min(), df["Factor_Pred_Return"].min(), df["Idio_Return"].min()],
        "Max": [df["NVDA"].max(), df["Factor_Pred_Return"].max(), df["Idio_Return"].max()],
        "Skew": [df["NVDA"].skew(), df["Factor_Pred_Return"].skew(), df["Idio_Return"].skew()],
        "Kurtosis": [df["NVDA"].kurtosis(), df["Factor_Pred_Return"].kurtosis(), df["Idio_Return"].kurtosis()],
        "Avg|Return|": [df["NVDA"].abs().mean(),
                        df["Factor_Pred_Return"].abs().mean(),
                        df["Idio_Return"].abs().mean()],
    }

    summary_df = pd.DataFrame(series_map)

    def highlight_vals(v):
        if isinstance(v, (int, float)):
            if v > 0: return "color:green; font-weight:600;"
            elif v < 0: return "color:red; font-weight:600;"
        return ""

    styled_summary = (
        summary_df.style
        .applymap(lambda _: "font-weight:bold;", subset=["Series"])
        .applymap(highlight_vals)
        .format({
            "Count": "{:d}",
            "Mean": "{:.4%}",
            "Volatility (Std Dev)": "{:.4%}",
            "Min": "{:.4%}",
            "Max": "{:.4%}",
            "Skew": "{:.2f}",
            "Kurtosis": "{:.2f}",
            "Avg|Return|": "{:.4%}",
        })
    )

    st.dataframe(styled_summary, use_container_width=True, hide_index=True)

    st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)


    # ---------------------------------------------------------------
    # 4️⃣ TWO MAIN CHARTS SIDE-BY-SIDE
    # ---------------------------------------------------------------
    st.markdown("<h3 style='font-size:17px;'>Return Decomposition — Daily & Cumulative</h3>",
                unsafe_allow_html=True)

    left, right = st.columns(2)

    # Daily decomposition chart
    with left:
        st.markdown("**Daily Returns: Total vs Systematic vs Idiosyncratic**")

        series_daily = [
            ("Total (NVIDIA)", df["NVDA"], 2.0, COLORS["blue"]),
            ("Systematic (Factors)", df["Factor_Pred_Return"], 1.6, COLORS["orange"]),
            ("Idiosyncratic", df["Idio_Return"], 1.6, COLORS["green"]),
        ]
        
        st.markdown("<p style='font-size:12px; line-height:1.3;'>This visual shows how NVIDIA’s daily returns split into factor-driven and stock-specific components, helping identify how much of each move was market-driven versus company-specific.</p>", unsafe_allow_html=True)

        fig_daily = plot_multitimeseries_with_event_lines(
            df["Date"], series_daily, df_e["EarningsDate"],
            "", "Return"
        )
        st.plotly_chart(fig_daily, use_container_width=True)

    # Cumulative decomposition chart
    with right:
        st.markdown("**Cumulative Returns: Total vs Systematic vs Idiosyncratic**")

        cum_total = df["NVDA"].cumsum()
        cum_factor = df["Factor_Pred_Return"].cumsum()
        cum_idio = df["Idio_Return"].cumsum()

        series_cum = [
            ("Total (Cumulative)", cum_total, 2.0, COLORS["blue"]),
            ("Systematic (Cumulative)", cum_factor, 1.6, COLORS["orange"]),
            ("Idiosyncratic (Cumulative)", cum_idio, 1.6, COLORS["green"]),
        ]
        
        st.markdown("<p style='font-size:12px; line-height:1.3;'>This chart tracks cumulative NVIDIA performance and decomposes how much of the long-term trend came from market factors versus idiosyncratic drivers.</p>", unsafe_allow_html=True)


        fig_cum = plot_multitimeseries_with_event_lines(
            df["Date"], series_cum, df_e["EarningsDate"],
            "", "Cumulative Return"
        )
        st.plotly_chart(fig_cum, use_container_width=True)

    st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)


    # ---------------------------------------------------------------
    # 5️⃣ RETURN DECOMPOSITION KPI TILES
    # ---------------------------------------------------------------
    st.markdown("<h3 style='font-size:17px;'>Factor Return Decomposition Key Statistics</h3>",
                unsafe_allow_html=True)

    tot, fac, idi, r2, corr_tf, hit_rate = get_model_decomposition_stats(df)

    k = st.columns(5)
    with k[0]: st.markdown(kpi_tile("R² Explained", r2, "📐"), unsafe_allow_html=True)
    with k[1]: st.markdown(kpi_tile("Corr(T,F)", corr_tf, "🔗"), unsafe_allow_html=True)
    with k[2]: st.markdown(kpi_tile("Hit Rate", hit_rate, "🎯"), unsafe_allow_html=True)
    with k[3]: st.markdown(kpi_tile("Avg |Idio|", idi.abs().mean(), "📊"), unsafe_allow_html=True)
    with k[4]: st.markdown(kpi_tile("Avg |Factor|", fac.abs().mean(), "📈"), unsafe_allow_html=True)

    st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)


    # ---------------------------------------------------------------
    # 6️⃣ TOP 5 FACTOR CONTRIBUTORS — SORTED BY TOTAL CONTRIBUTION
    # ---------------------------------------------------------------
    st.markdown("<h3 style='font-size:17px;'>Top 5 Factor Contributors</h3>",
                unsafe_allow_html=True)
    st.markdown("<p style='font-size:12px; line-height:1.3;'>This table ranks which market factors contributed the most to NVIDIA’s returns over the selected period.</p>", unsafe_allow_html=True)

    contrib_cols = [c for c in df.columns if c.endswith("_contrib")]

    if not contrib_cols:
        st.info("No factor contribution columns available.")
    else:
        contrib_sums = df[contrib_cols].sum(axis=0)           
        contrib_abs = df[contrib_cols].abs().sum(axis=0)      
        fac_total_sum = df.get("Factor_Pred_Return", pd.Series()).sum()

        # SORT BY TOTAL CONTRIBUTION, NOT ABSOLUTE
        top5_idx = contrib_sums.sort_values(ascending=False).head(5).index

        top5 = pd.DataFrame({
            "Factor": [i.replace("_contrib", "") for i in top5_idx],
            "Total Contribution": contrib_sums.loc[top5_idx].values,
            "Total |Contribution|": contrib_abs.loc[top5_idx].values,
            "% of Factor-Pred Total": [
                np.nan if fac_total_sum == 0 else contrib_sums.loc[i] / fac_total_sum
                for i in top5_idx
            ],
            "Avg Daily Contribution": df[top5_idx].mean().values,
            "Contribution Volatility (Std)": df[top5_idx].std().values,
        })

        def color_vals(v):
            if isinstance(v, (int, float)):
                if v > 0: return "color:green;"
                elif v < 0: return "color:red;"
            return ""


        styled_top5 = (
            top5.style
            .applymap(lambda _: "font-weight:bold;", subset=["Factor"])
            .applymap(color_vals)
            .format({
                "Total Contribution": "{:.4%}",
                "Total |Contribution|": "{:.4%}",
                "% of Factor-Pred Total": "{:.1%}",
                "Avg Daily Contribution": "{:.4%}",
                "Contribution Volatility (Std)": "{:.4%}",
            })
        )

        st.dataframe(styled_top5, use_container_width=True, hide_index=True)

    st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)


    # ---------------------------------------------------------------
    # 7️⃣ FACTOR CORRELATION HEATMAP — MOVED TO END
    # ---------------------------------------------------------------
    st.markdown("<h3 style='font-size:17px;'>Factor Return Correlation Heatmap</h3>",
                unsafe_allow_html=True)
    st.markdown("<p style='font-size:12px; line-height:1.3;'>Shows how selected factors move together, helping identify correlation clusters and potential redundancy.</p>", unsafe_allow_html=True)

    if len(fac_ret_cols := [f"{f}_ret" for f in selected_factors if f"{f}_ret" in df.columns]) >= 2:
        corr = df[fac_ret_cols].corr()
        fig = go.Figure(go.Heatmap(
            z=corr.values,
            x=[c.replace("_ret", "") for c in fac_ret_cols],
            y=[c.replace("_ret", "") for c in fac_ret_cols],
            colorscale="RdBu", zmin=-1, zmax=1,
        ))
        fig = apply_plotly_layout(fig,
                                  title="",
                                  bottom=60)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Select 2 or more factors to view correlations.")
