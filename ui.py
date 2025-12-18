"""
ui.py
Author: Nawaz Pasha

Purpose:
- Provides global CSS styling
- Provides sidebar branding block
- Provides safe dataframe display helper for Streamlit
"""

import os
import streamlit as st
import pandas as pd
from typing import Optional
import numpy as np


# =====================================================================
# GLOBAL CSS
# =====================================================================

def use_global_css():
    """
    Injects global CSS for tabs, sidebar, and plot aesthetics.
    """
    st.markdown(
        """
        <style>
        /* Sidebar tidy */
        section[data-testid="stSidebar"] pre,
        section[data-testid="stSidebar"] code {
            display: none !important;
        }

        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
            margin-bottom: .5rem;
        }

        /* Tabs: nicely wrapped multi-line labels */
        .stTabs [data-baseweb="tab-list"] {
            flex-wrap: wrap !important;
            row-gap: 4px !important;
        }
        .stTabs [data-baseweb="tab"] {
            height: auto !important;
            white-space: normal !important;
            line-height: 1.2 !important;
            padding: 6px !important;
            margin-right: 6px !important;
            border-radius: 6px !important;
        }
        .stTabs [data-baseweb="tab"] p {
            margin: 0 !important;
        }
        .stTabs [role="tablist"]::-webkit-scrollbar {
            display: none !important;
        }

        /* Plotly modebar to top layer */
        .modebar { z-index: 10 !important; }

        </style>
        """,
        unsafe_allow_html=True,
    )


# =====================================================================
# SIDEBAR BRANDING
# =====================================================================

def control_panel_branding():
    """
    Renders NVIDIA branding in Streamlit sidebar.
    """
    with st.sidebar:
        with st.container():
            logo_path = "nvidia_logo.png"
            if os.path.exists(logo_path):
                st.image(logo_path)
            else:
                st.markdown("### **NVIDIA**")

        st.markdown("---")
        st.markdown("**Developer - Nawaz Pasha**")
        st.markdown("📞 +91-9986551895")
        st.markdown("✉️ Navvu18@gmail.com")
        st.markdown("---")


# =====================================================================
# SAFE DATAFRAME RENDERING
# =====================================================================

def df_show(df: pd.DataFrame, fmt: Optional[dict] = None):
    """
    Robust dataframe display wrapper.

    Fixes issues:
    - Styling crashes when df contains mixed types
    - Date columns displayed more cleanly
    - Formatting applied only to numeric columns

    Parameters:
    - df : DataFrame
    - fmt : {column: format_string}

    Returns:
    - Streamlit dataframe widget
    """

    df_clean = df.copy()

    # Convert datetime columns for safer rendering
    for col in df_clean.columns:
        if pd.api.types.is_datetime64_any_dtype(df_clean[col]):
            df_clean[col] = df_clean[col].dt.strftime("%Y-%m-%d")

    # Apply style formatting only if fmt provided
    if fmt:
        try:
            styled = df_clean.style.format(fmt)
            st.dataframe(styled, use_container_width=True)
        except Exception:
            # Fallback if Styler fails
            st.dataframe(df_clean, use_container_width=True)
    else:
        st.dataframe(df_clean, use_container_width=True)


def excel_style(df):
    def fmt(x):
        if isinstance(x, (float, int)):
            return f"{x:.2%}" if -1 <= x <= 1 else f"{x:.2f}"
        return x

    return (
        df.style
        .format(fmt)
        .set_table_styles(
            [
                {"selector": "th",
                 "props": "background-color:#f0f0f0; font-weight:bold; text-align:center; "
                          "padding:5px; border:1px solid #c8c8c8; font-size:12px;"},
                {"selector": "td",
                 "props": "padding:4px; border:1px solid #d9d9d9; font-size:12px;"},
                {"selector": "tbody tr:nth-child(even)",
                 "props": "background-color:#fafafa;"}
            ],
            overwrite=False
        )
        .set_properties(**{"text-align": "center"})
    )

