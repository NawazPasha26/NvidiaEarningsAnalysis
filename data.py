"""
@author - Nawaz Pasha 
data import update
"""
#data.py
#This module reads in the data input data from the csv files
#Also checks for formattings specially dates and updates them as required

import streamlit as st
import pandas as pd
import pathlib

def _require_file(path:str, friendly_name: str):
    if not pathlib.Path(path).exists():
        st.error(f"Missing required file: **{friendly_name}** ('{path}'). "
                 f"Place it in the same folder as 'app.py' and try again.")
        st.stop()
        
@st.cache_data(show_spinner=False)
def load_input_data():
    #Verify files exist before reading
    _require_file("01_case_study_returns.csv", "Returns CSV")
    _require_file("02_case_study_factor_loadings.csv", "Factor Loadings CSV")
    _require_file("03_case_study_earnings_dates.csv", "Earnings Dates CSV")
    
    r = pd.read_csv("01_case_study_returns.csv", header=2)
    l = pd.read_csv("02_case_study_factor_loadings.csv", header=2)
    raw_e = open("03_case_study_earnings_dates.csv", "r", encoding="utf-8").read().splitlines()
    #parse earnings dates rhobustly
    earn_dates = []
    for line in (ln.strip() for ln in raw_e):
        if "/" in line and any(ch.isdigit() for ch in line):
            for fmt in ("%m/%d/%Y", None):
                try:
                    earn_dates.append(pd.to_datetime(line, format=fmt) if fmt else pd.to_datetime(line))
                    break
                except Exception:
                    pass
    df_e = pd.DataFrame({"EarningsDate": sorted(set(earn_dates))})
    r.rename(columns={r.columns[0]: "Date"}, inplace=True)
    l.rename(columns={l.columns[0]: "Date"}, inplace=True)
    r["Date"], l["Date"] = pd.to_datetime(r["Date"], errors="coerce"), pd.to_datetime(l["Date"], errors="coerce")
    r = r.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    l = l.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    start, end = max(r["Date"].min(), l["Date"].min()), min(r["Date"].max(), l["Date"].max())
    all_factors = [c for c in r.columns if c not in ["Date", "NVDA"] and c in l.columns]
    return r, l, df_e, all_factors, pd.Timestamp(start), pd.Timestamp(end)


