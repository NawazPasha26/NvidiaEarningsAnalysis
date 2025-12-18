"""
Main Application Entry Point — NVIDIA Earnings Analysis Dashboard
================================================================

Author      : Nawaz Pasha
Last Updated: Thu Dec 18, 2025
File        : app.py

Overview
--------
This file is the **main entry point** for the Streamlit-based NVIDIA Earnings
Analysis dashboard covering the period 2022–2025.

Responsibilities of this module:
- Configure the Streamlit page layout and global styling
- Load and validate all input data sources
- Define and manage sidebar user controls (dates, factors, windows)
- Build the factor-return model and earnings event windows
- Create and route Streamlit tabs
- Delegate all tab-specific rendering to modular tab components

Architecture
------------
The application follows a **modular, separation-of-concerns design**:

- app.py
    → Orchestration layer only (no heavy analytics, no plotting logic)
- analysis.py
    → All statistical, econometric, and quantitative computations
- plots.py
    → All Plotly-based visualization helpers
- ui.py
    → Global CSS, branding, and reusable UI components
- data.py
    → Data loading and preprocessing utilities
- constants.py
    → Color palettes, text blocks, and configuration constants
- tabs/
    → One module per tab:
        • overview.py
        • event_analysis.py
        • volatility_statistics.py
        • summary.py

Execution Flow
--------------
1. Load global CSS and dashboard branding
2. Load input datasets (returns, factors, earnings dates)
3. Initialize sidebar controls and validate user selections
4. Build return decomposition model and earnings event windows
5. Create Streamlit tabs
6. Render each tab via its dedicated render_* function

Design Principles
-----------------
- app.py contains **no business logic** and **no plotting logic**
- Each tab is independently maintainable and testable
- Data is built once and passed downstream to all tabs
- All UI behavior is deterministic and driven by user inputs
- Optimized for readability, scalability, and future extension

Usage
-----
Run the application with:
    streamlit run app.py

This file should remain lightweight and should only grow when:
- A new tab is added
- A new global control is introduced
- The application startup flow changes
"""



#import os
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
#from typing import Optional
from ui import use_global_css, control_panel_branding
from data import load_input_data
from analysis import (build_return_model, build_event_window)
from overview import render_overview
from event_analysis import render_event_analysis
from volatility_statistics import render_volatility_statistics
from summary import render_summary

st.set_page_config(page_title="NVIDIA Earnings Analysis", layout="wide")
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except Exception:
    plt.style.use("classic")



# ---- App start ----
use_global_css()
control_panel_branding()
r, l, e, all_factors, global_start, global_end = load_input_data()


# Sidebar controls
st.sidebar.header("Filters")
# Inform user about available date range
st.sidebar.markdown(
    f"Input Return Data is available only from {global_start.strftime('%b %Y')} to {global_end.strftime('%b %Y')} "
)
# Default date toggle
use_default_dates = st.sidebar.checkbox("Use dataset default Date Range (Oct 2022 - Sep 2025)", value=True)
# Prepare set of valid dates from the loaded return series 'r'
_valid_dates = set(r["Date"].dt.date)

if use_default_dates:
    sel_start = global_start
    sel_end = global_end
else:
    # Individual start and end date pickers
    try:
        start_date = st.sidebar.date_input(
            "Start date", global_start.date(),
            min_value=global_start.date(), max_value=global_end.date(), key="start_date"
        )
        end_date = st.sidebar.date_input(
            "End date", global_end.date(),
            min_value=global_start.date(), max_value=global_end.date(), key="end_date"
        )

        s_ts = pd.Timestamp(start_date)
        e_ts = pd.Timestamp(end_date)

        # Basic ordering validation
        if s_ts > e_ts:
            st.sidebar.error("Start date must be on or before End date. Please select proper valid dates.")
            sel_start, sel_end = global_start, global_end
        else:
            # Check that the exact selected dates exist in the input data (trade dates)
            if (s_ts.date() not in _valid_dates) or (e_ts.date() not in _valid_dates):
                # Show a friendly, prominent message in the main area (not only sidebar)
                st.error(
                    "Please select valid dates — the return input data contains trading dates only from "
                    "September-2022 to September-2025. Pick start/end dates that are present in the dataset."
                )
                st.sidebar.error("Selected start or end date is not present in the input data. Reverting to defaults.")
                sel_start, sel_end = global_start, global_end
            else:
                sel_start, sel_end = s_ts, e_ts
    except Exception:
        st.sidebar.error("Invalid date input. Using dataset default date range.")
        sel_start, sel_end = global_start, global_end
        
selected_factors = st.sidebar.multiselect(
    "Factors (affect factor-predicted return)", options=all_factors, default=all_factors
)
win = st.sidebar.slider("Event window (days before/after)", 5, 20, 10, 1)
roll = st.sidebar.slider("Rolling volatility window (days)", 10, 60, 20, 5)

# Return components filter
component_options = ["Total", "Systematic", "Idiosyncratic"]
#show_components = st.sidebar.multiselect("Return components", component_options, default=component_options)
show_components=component_options


# Build model
df, selected_factors, df_e = build_return_model(r, l, e, selected_factors, sel_start, sel_end)


# Ensure df is strictly within selected range
df = df[(df["Date"] >= sel_start) & (df["Date"] <= sel_end)].copy()

# Ensure earnings used are ONLY those inside df date range
df_e = df_e[
    (df_e["EarningsDate"] >= df["Date"].min()) &
    (df_e["EarningsDate"] <= df["Date"].max())
].copy()

df.reset_index(drop=True, inplace=True)
df_e.reset_index(drop=True, inplace=True)

ev_out = build_event_window(df, df_e, window=win)


# Tabs
t1,t2, t3, t4= st.tabs([
    "Overview", "Earnings Event Analysis",
    "Earnings Volatility & Statistical Reliability",
    "Summary"
])


# =====================================================================
# Start Tab Rendering

# =====================================================================
#'''Start Rendering all 4 Tabs - "Overview", "Earnings Event Analysis",
#"Earnings Volatility & Statistical Reliability",
#"Summary"'''
# =====================================================================


with t1:
    render_overview(df, df_e, selected_factors, win)

with t2:
    render_event_analysis(df, df_e, ev_out, win)
    
with t3:
    render_volatility_statistics(df, df_e, ev_out, win, roll)
    
with t4:
    render_summary(df, df_e, ev_out, win)


#'''End of all Tabs Rendering on the Streamlit Dashboard'''

# End Tabs Rendering
# =====================================================================

