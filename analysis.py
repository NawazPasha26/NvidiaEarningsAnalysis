"""
analysis.py
Author: Nawaz Pasha

This module contains:
- Return model decomposition
- Event window construction
- Event-study aggregated analytics
- All required helper functions for the New Event Analysis Tab
"""

import numpy as np
import pandas as pd


# =====================================================================
# 1. RETURN MODEL DECOMPOSITION
# =====================================================================

def build_return_model(r, l, e, selected_factors, date_start, date_end):
    """
    Build factor-predicted and idiosyncratic return series.
    """
    r_ = r[(r["Date"] >= date_start) & (r["Date"] <= date_end)].copy()
    l_ = l[(l["Date"] >= date_start) & (l["Date"] <= date_end)].copy()

    selected_factors = [c for c in selected_factors if c in r_.columns and c in l_.columns]

    if selected_factors:
        df = pd.merge(
            r_[["Date", "NVDA"] + selected_factors],
            l_[["Date"] + selected_factors],
            on="Date", suffixes=("_ret", "_beta"),
        )
        beta = df[[f"{f}_beta" for f in selected_factors]].to_numpy()
        ret = df[[f"{f}_ret" for f in selected_factors]].to_numpy()
        df["Factor_Pred_Return"] = (beta * ret).sum(axis=1)
        for f in selected_factors:
            df[f"{f}_contrib"] = df[f"{f}_beta"] * df[f"{f}_ret"]
    else:
        df = r_[["Date", "NVDA"]].copy()
        df["Factor_Pred_Return"] = 0.0

    df["Idio_Return"] = df["NVDA"] - df["Factor_Pred_Return"]
    df["NVDA_Cum"] = (1 + df["NVDA"]).cumprod() - 1

    return df, selected_factors, e.copy()


# =====================================================================
# 2. EVENT-WINDOW CONSTRUCTION
# =====================================================================

def get_next_trading_day(date_series: pd.Series, target_date: pd.Timestamp):
    idx = date_series.searchsorted(target_date, side="right")
    return date_series.iloc[idx] if idx < len(date_series) else None


def build_event_window(df, df_e, window=10):
    dates = df["Date"].reset_index(drop=True)
    valid, centers = [], []

    for d in df_e["EarningsDate"].dropna():
        d0 = get_next_trading_day(dates, d)
        if d0 is None:
            continue

        i0 = int(dates.searchsorted(d0))
        lo, hi = i0 - window, i0 + window

        if lo < 0 or hi >= len(dates):
            continue

        valid.append((lo, i0, hi))
        centers.append(i0)

    if not valid:
        return None

    idxs = np.arange(-window, window + 1)

    mats = {
        col: np.vstack([df[col].iloc[lo:hi+1].to_numpy() for (lo, i0, hi) in valid])
        for col in ["NVDA", "Factor_Pred_Return", "Idio_Return"]
    }

    info = pd.DataFrame({
        "event_center_index": centers,
        "event_date": [df.loc[i0, "Date"] for (_, i0, _) in valid]
    })

    return mats, idxs, info


# =====================================================================
# 3. SUPPORT FUNCTIONS FOR DECOMPOSITION
# =====================================================================

def safe_var(x):
    v = np.nanvar(x, ddof=1)
    return np.nan if v == 0 else v


def get_model_decomposition_stats(df):
    tot, fac, idi = df["NVDA"], df["Factor_Pred_Return"], df["Idio_Return"]
    sv_tot, sv_fac = safe_var(tot), safe_var(fac)

    r2 = np.nan if sv_tot in [0, np.nan] else float(sv_fac / sv_tot)
    corr_tf = float(np.corrcoef(tot.fillna(0), fac.fillna(0))[0, 1]) if len(df) > 1 else np.nan
    hit_rate = float(np.mean(np.sign(tot) == np.sign(fac))) if len(df) > 0 else np.nan

    return tot, fac, idi, r2, corr_tf, hit_rate


# =====================================================================
# 4. NEW EVENT ANALYSIS — WINDOW-LEVEL METRICS
# =====================================================================

# 4.1 Interval return: compounded
def compute_interval_move(df, df_e, window):
    dates = df["Date"].reset_index(drop=True)
    out = []

    for ed in df_e["EarningsDate"].dropna():
        d0 = get_next_trading_day(dates, ed)
        if d0 is None:
            continue

        i0 = int(dates.searchsorted(d0))
        lo = i0 - window
        hi = i0 + window

        if lo < 0 or hi >= len(dates):
            continue

        sub = df.iloc[lo:hi+1]["NVDA"].dropna()
        if len(sub) == 0:
            interval = np.nan
        else:
            interval = (1 + sub).prod() - 1

        out.append({
            "EarningsDate": ed,
            "StartDate": dates.loc[lo],
            "EndDate": dates.loc[hi],
            "IntervalReturn": interval
        })

    return pd.DataFrame(out)


# 4.2 Progressive cumulative curves (average + median)
def compute_progressive_curves(mats, idxs):
    nv = mats["NVDA"]  # shape: (events, T)

    avg_curve = np.nanmean(nv, axis=0)
    med_curve = np.nanmedian(nv, axis=0)

    # Compound cumulative returns (quant-correct)
    def cum_comp(r):
        return (1 + r).cumprod() - 1

    return {
        "avg": cum_comp(avg_curve),
        "median": cum_comp(med_curve),
        "idxs": idxs
    }


# 4.3 Pattern table (before/after/custom buckets)
def extract_pattern_table_values(df, df_e, window):
    dates = df["Date"].reset_index(drop=True)
    out = []

    for ed in df_e["EarningsDate"].dropna():
        d0 = get_next_trading_day(dates, ed)
        if d0 is None:
            continue

        i0 = int(dates.searchsorted(d0))
        lo = i0 - window
        hi = i0 + window
        if lo < 0 or hi >= len(dates):
            continue

        row = {"EarningsDate": ed}

        def cum_move(i_start, i_end):
            if i_start < 0 or i_end >= len(df):
                return np.nan
            sub = df.iloc[i_start:i_end+1]["NVDA"].dropna()
            return (1 + sub).prod() - 1 if len(sub) > 0 else np.nan

        # BEFORE (fixed buckets)
        row["2W_Before"] = cum_move(i0-10, i0-1)
        row["1W_Before"] = cum_move(i0-5, i0-1)
        row["3D_Before"] = cum_move(i0-3, i0-1)
        row["2D_Before"] = cum_move(i0-2, i0-1)
        row["1D_Before"] = cum_move(i0-1, i0-1)

        # EVENT DAY
        row["Day0"] = df.loc[i0, "NVDA"]

        # AFTER
        row["1D_After"] = cum_move(i0+1, i0+1)
        row["2D_After"] = cum_move(i0+1, i0+2)
        row["3D_After"] = cum_move(i0+1, i0+3)
        row["1W_After"] = cum_move(i0+1, i0+5)
        row["2W_After"] = cum_move(i0+1, i0+10)

        # CUSTOM interval: −window → +window
        row["IntervalRange"] = f"{dates.loc[lo].date()} → {dates.loc[hi].date()}"
        row["CustomInterval"] = cum_move(lo, hi)

        out.append(row)

    return pd.DataFrame(out)


# =====================================================================
# 5. SUMMARY ROWS (A, B, C)
# =====================================================================

def summarize_pattern_table(df_pat):
    cols = [c for c in df_pat.columns if c not in ["EarningsDate", "IntervalRange"]]

    A = pd.DataFrame({"Metric": ["Avg Return", "Pos Occ %", "Neg Occ %"]})
    avg_vals = df_pat[cols].mean()
    pos = (df_pat[cols] > 0).mean()
    neg = (df_pat[cols] < 0).mean()

    for c in cols:
        A[c] = [avg_vals[c], pos[c], neg[c]]

    B = pd.DataFrame({"Metric": ["Abs Avg Return", "Max Abs Return", "Min Abs Return"]})
    abs_avg = df_pat[cols].abs().mean()
    max_abs = df_pat[cols].abs().max()
    min_abs = df_pat[cols].abs().min()

    for c in cols:
        B[c] = [abs_avg[c], max_abs[c], min_abs[c]]

    C = pd.DataFrame({"Metric": ["Median Return", "Max Positive Return", "Max Negative Return"]})
    med = df_pat[cols].median()
    max_pos = df_pat[cols].max()
    max_neg = df_pat[cols].min()

    for c in cols:
        C[c] = [med[c], max_pos[c], max_neg[c]]

    return {"A": A, "B": B, "C": C}


# =====================================================================
# 6. STATISTICS TABLE (Positive / Negative / Total)
# =====================================================================

def compute_stat_table(interval_returns):
    s = interval_returns.dropna()
    pos = s[s > 0]
    neg = s[s < 0]

    def stats_block(x):
        if len(x) == 0:
            return {
                "Occurrences %": np.nan,
                "Average": np.nan,
                "Median": np.nan,
                "Absolute Average": np.nan,
                "High": np.nan,
                "Low": np.nan,
                "Std Dev": np.nan,
            }
        return {
            "Occurrences %": len(x) / len(s),
            "Average": x.mean(),
            "Median": x.median(),
            "Absolute Average": x.abs().mean(),
            "High": x.max(),
            "Low": x.min(),
            "Std Dev": x.std(ddof=1),
        }

    st_pos = stats_block(pos)
    st_neg = stats_block(neg)
    st_tot = stats_block(s)

    return pd.DataFrame({
        "Stat": list(st_tot.keys()),
        "Positive": list(st_pos.values()),
        "Negative": list(st_neg.values()),
        "Total": list(st_tot.values()),
    })


# =====================================================================
# 7. TOP/BOTTOM 10 TABLES
# =====================================================================

def compute_top_bottom(interval_df, n=5):
    """
    Return top N positive and top N negative interval returns.
    Only strictly positive values appear in the positives list.
    Only strictly negative values appear in the negatives list.
    """

    df = interval_df.copy()

    # Create range label
    df["Range"] = (
        df["StartDate"].dt.strftime("%Y-%m-%d")
        + " → "
        + df["EndDate"].dt.strftime("%Y-%m-%d")
    )

    # Ensure IntervalReturn is numeric
    df = df.dropna(subset=["IntervalReturn"])

    # ---- STRICT FILTERS ----
    df_pos = df[df["IntervalReturn"] > 0]      # strictly positive
    df_neg = df[df["IntervalReturn"] < 0]      # strictly negative

    # ---- SELECT TOP N ----
    top_pos = df_pos.sort_values(
        "IntervalReturn", ascending=False
    ).head(n)

    top_neg = df_neg.sort_values(
        "IntervalReturn", ascending=True
    ).head(n)

    return top_pos, top_neg



def mask_triplet(idxs):
    return (idxs < 0), (idxs == 0), (idxs > 0)



# =====================================================================
# 8. RISK-RATIO TABLE
# =====================================================================

def compute_risk_ratios(interval_returns):
    s = interval_returns.dropna()
    down = s[s < 0]

    metrics = {
        "Biggest Historical Down Move": abs(down.min()) if len(down) else np.nan,
        "Avg Down Move": abs(down.mean()) if len(down) else np.nan,
        "Std Dev": s.std(ddof=1) if len(s) > 1 else np.nan,
        "Std Dev of Down Moves": down.std(ddof=1) if len(down) > 1 else np.nan,
    }

    avg_ret = s.mean()
    med_ret = s.median()

    out = []
    for m, v in metrics.items():
        if v in [0, np.nan]:
            out.append([m, np.nan, np.nan])
        else:
            out.append([m, avg_ret / v, med_ret / v])

    return pd.DataFrame(out, columns=["Risk Metric", "Avg Return / Metric", "Median Return / Metric"])

def compute_event_averages(ev_out):
    mats, idxs2, info = ev_out

    avg_total = np.nanmean(mats["NVDA"], axis=0)
    avg_factor = np.nanmean(mats["Factor_Pred_Return"], axis=0)
    avg_idio = np.nanmean(mats["Idio_Return"], axis=0)

    avg_dict = {
        "total": avg_total,
        "factor": avg_factor,
        "idio": avg_idio
    }

    return idxs2, avg_dict

def mean_over_mask(ts, m):
    return float(np.nanmean(ts[m]))

def compute_pre_post_means(avg, idxs):
    pre, ev, post = mask_triplet(idxs)
    a = avg
    return (
        mean_over_mask(a["total"], pre), mean_over_mask(a["total"], ev), mean_over_mask(a["total"], post),
        mean_over_mask(a["factor"], pre), mean_over_mask(a["factor"], ev), mean_over_mask(a["factor"], post),
        mean_over_mask(a["idio"], pre), mean_over_mask(a["idio"], ev), mean_over_mask(a["idio"], post)
    )


def compute_pre_post_abs_change(mats, idxs):
    pre, _, post = mask_triplet(idxs)
    pre_abs = np.nanmean(np.abs(mats["NVDA"][:, pre]), axis=1)
    post_abs = np.nanmean(np.abs(mats["NVDA"][:, post]), axis=1)
    return pre_abs, post_abs, (post_abs - pre_abs)


def build_event_comparison_table(df, df_e, window):
    dates = df["Date"].reset_index(drop=True)
    returns = df["NVDA"].reset_index(drop=True)

    def find_reaction_day(ed):
        idx = dates.searchsorted(ed, side="right")
        return idx if idx < len(dates) else None

    def cum_return(lo, hi):
        if lo < 0 or hi >= len(returns): 
            return np.nan
        sub = returns.iloc[lo:hi+1].dropna()
        return (1 + sub).prod() - 1 if len(sub) > 0 else np.nan

    def drift(lo, hi):
        return cum_return(lo, hi)

    def vol(lo, hi):
        if lo < 0 or hi >= len(returns): 
            return np.nan
        sub = returns.iloc[lo:hi+1].dropna()
        return sub.std(ddof=1) if len(sub) > 1 else np.nan

    def pos_neg_stats(lo, hi):
        if lo < 0 or hi >= len(returns): 
            return (np.nan, np.nan, np.nan, np.nan, np.nan)
        sub = returns.iloc[lo:hi+1].dropna()
        if len(sub) == 0:
            return (np.nan, np.nan, np.nan, np.nan, np.nan)
        pos = sub[sub > 0]
        neg = sub[sub < 0]
        return (
            len(pos)/len(sub),
            len(neg)/len(sub),
            pos.mean() if len(pos) else np.nan,
            neg.mean() if len(neg) else np.nan,
            max(abs(pos.max() if len(pos) else np.nan), abs(neg.min() if len(neg) else np.nan))
        )

    metrics = [
        "Day 0 Return",
        f"Full Interval Return (±{window})",
        f"Pre-event Drift (-{window} to -1)",
        f"Post-event Drift (+1 to +{window})",
        "Volatility (±window)",
        "% Positive Days",
        "% Negative Days",
        "Avg Positive Move",
        "Avg Negative Move",
        "Max Absolute Move",
        "Cumulative Return (±window)",
        "Max Drawdown (±window)",
        "Max Run-up (±window)"
    ]

    table = {m: [] for m in metrics}
    event_labels = []

    for ed in df_e["EarningsDate"].dropna():
        i0 = find_reaction_day(ed)
        if i0 is None:
            for m in metrics:
                table[m].append(np.nan)
            event_labels.append(str(ed.date()))
            continue

        lo, hi = i0 - window, i0 + window

        day0 = returns.iloc[i0]
        full_interval = cum_return(lo, hi)
        pre_drift = drift(i0 - window, i0 - 1)
        post_drift = drift(i0 + 1, i0 + window)
        volatility = vol(lo, hi)

        pos_pct, neg_pct, avg_up, avg_down, max_abs = pos_neg_stats(lo, hi)
        cumulative = full_interval

        sub = (1 + returns.iloc[lo:hi+1]).cumprod() - 1
        max_dd = sub.min()
        max_ru = sub.max()

        table["Day 0 Return"].append(day0)
        table[f"Full Interval Return (±{window})"].append(full_interval)
        table[f"Pre-event Drift (-{window} to -1)"].append(pre_drift)
        table[f"Post-event Drift (+1 to +{window})"].append(post_drift)
        table["Volatility (±window)"].append(volatility)
        table["% Positive Days"].append(pos_pct)
        table["% Negative Days"].append(neg_pct)
        table["Avg Positive Move"].append(avg_up)
        table["Avg Negative Move"].append(avg_down)
        table["Max Absolute Move"].append(max_abs)
        table["Cumulative Return (±window)"].append(cumulative)
        table["Max Drawdown (±window)"].append(max_dd)
        table["Max Run-up (±window)"].append(max_ru)

        event_labels.append(str(ed.date()))

    # FIXED — Build DataFrame using correct orientation
    df_out = pd.DataFrame(table).T
    df_out.index = metrics
    df_out.columns = event_labels
    return df_out



def bootstrap_distribution(data, stat_fn, n_boot=20000, seed=42):
    """
    Generic bootstrap engine.
    data: 1D array-like
    stat_fn: function applied to each bootstrap sample
    returns: array of bootstrap statistics
    """
    data = np.array(data).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(data), size=(n_boot, len(data)))
    samples = data[idx]
    stats = stat_fn(samples, axis=1)
    return stats

def bootstrap_ci_pvalue(data, observed_value, stat_fn, ci_level=0.95, n_boot=20000):
    """
    Computes:
    - bootstrap CI
    - bootstrap p-value (two-sided)
    """
    dist = bootstrap_distribution(data, stat_fn, n_boot=n_boot)

    lower = np.percentile(dist, (1-ci_level)*50)
    upper = np.percentile(dist, 100 - (1-ci_level)*50)

    # two-sided p-value
    diffs = np.abs(dist - np.mean(dist))
    obs = np.abs(observed_value - np.mean(dist))
    p_val = np.mean(diffs >= obs)

    return (lower, upper), p_val

def bootstrap_all_tests(pre, post, day0, ci_level=0.95, n_boot=20000):
    """
    Returns a dict with:
    - Mean test
    - Median test
    - Volatility test (std)
    - Positive% test
    For: Pre-window, Post-window, and Day-0.

    All statistics include:
    {
        'observed': xx,
        'ci': (lower, upper),
        'p': p_value
    }

    Includes SAFE HANDLING for:
    - Empty pre or post windows
    - Unequal sample sizes
    - Avoiding invalid (post - pre) broadcasting errors
    """

    results = {}

    # -------------------------------
    # REQUIRED: Convert to numpy arrays
    # -------------------------------
    pre = np.array(pre)
    post = np.array(post)
    day0 = np.array([day0])  # ensure array for consistency

    stats_to_run = {
        "Mean": np.mean,
        "Median": np.median,
        "Volatility (Std)": np.std,
        "% Positive": lambda x: np.mean(x > 0)
    }

    for name, fn in stats_to_run.items():

        # ============================================================
        # SAFE PRE–POST DIFFERENCE TEST
        # ============================================================
        if len(pre) > 0 and len(post) > 0:

            # Align sizes to avoid broadcasting failures
            m = min(len(pre), len(post))
            pre_s = pre[:m]
            post_s = post[:m]

            observed_diff = float(fn(post_s) - fn(pre_s))

            # Bootstrap difference distribution
            boot_diff = np.zeros(n_boot)
            for i in range(n_boot):
                pre_bs = np.random.choice(pre_s, size=m, replace=True)
                post_bs = np.random.choice(post_s, size=m, replace=True)
                boot_diff[i] = fn(post_bs) - fn(pre_bs)

            # CI
            ci_diff = (
                float(np.percentile(boot_diff, (1 - ci_level) * 50)),
                float(np.percentile(boot_diff, 100 - (1 - ci_level) * 50))
            )

            # Two-sided bootstrap p-value
            p_diff = float(
                np.mean(
                    np.abs(boot_diff - boot_diff.mean()) >=
                    np.abs(observed_diff - boot_diff.mean())
                )
            )

        else:
            # Not enough data → return NaNs instead of crashing
            observed_diff = np.nan
            ci_diff = (np.nan, np.nan)
            p_diff = np.nan

        # ============================================================
        # DAY 0 TEST — always uses PRE window as null distribution
        # ============================================================
        if len(pre) > 0:
            observed_day0 = float(fn(day0))

            boot_day0 = np.zeros(n_boot)
            for i in range(n_boot):
                pre_bs = np.random.choice(pre, size=len(pre), replace=True)
                boot_day0[i] = fn(pre_bs)

            ci_d0 = (
                float(np.percentile(boot_day0, (1 - ci_level) * 50)),
                float(np.percentile(boot_day0, 100 - (1 - ci_level) * 50))
            )

            p_d0 = float(
                np.mean(
                    np.abs(boot_day0 - boot_day0.mean()) >=
                    np.abs(observed_day0 - boot_day0.mean())
                )
            )

        else:
            observed_day0 = np.nan
            ci_d0 = (np.nan, np.nan)
            p_d0 = np.nan

        # ============================================================
        # STORE RESULTS
        # ============================================================
        results[name] = {
            "day0": {
                "observed": observed_day0,
                "ci": ci_d0,
                "p": p_d0
            },
            "diff": {
                "observed": observed_diff,
                "ci": ci_diff,
                "p": p_diff
            }
        }

    return results


# ==============================================================
# BOOTSTRAP VOLATILITY TESTS (ABSOLUTE RETURNS) — FIXED FOR IDX STRUCTURE
# ==============================================================


def bootstrap_ci_pvalue_vol(null_sample, observed_value, ci_level, n_boot):
    """
    Bootstrap CI and p-value for absolute-return volatility tests.
    Ensures null_sample is always 1-D.
    """

    # --- CRITICAL FIX: force 1-D vector regardless of input shape ---
    null_sample = np.asarray(null_sample).flatten()

    if null_sample.size == 0:
        return (np.nan, np.nan), np.nan

    n = len(null_sample)
    boot_vals = np.zeros(n_boot)

    for i in range(n_boot):
        resample = np.random.choice(null_sample, size=n, replace=True)
        boot_vals[i] = np.mean(np.abs(resample))

    alpha = 1 - ci_level
    ci_low = np.percentile(boot_vals, 100 * alpha / 2)
    ci_high = np.percentile(boot_vals, 100 * (1 - alpha / 2))

    p_val = np.mean(
        np.abs(boot_vals - np.mean(boot_vals)) >=
        np.abs(observed_value - np.mean(boot_vals))
    )

    return (ci_low, ci_high), p_val


def bootstrap_vol_tests(mats, idxs, ci_level=0.95, n_boot=20000):
    """
    Bootstrap significance tests for ABSOLUTE RETURN VOLATILITY:
    • Day-0 |return|
    • Pre-window vs Post-window mean |return|
    Works for:
        - Total (NVDA)
        - Idiosyncratic (Idio_Return)
    FIXED: uses mask_triplet(idxs) instead of idxs["pre"], idxs["post"].
    """

    # --------------------------------------------------------------
    # 1. Build boolean masks for pre / day0 / post windows
    # --------------------------------------------------------------
    pre_mask, day0_mask, post_mask = mask_triplet(idxs)

    def extract_abs(mat, mask):
        # Flatten all events × mask_len into 1-D vector
        out = mat[:, mask].astype(float).flatten()
        return out[~np.isnan(out)]

    # --------------------------------------------------------------
    # 2. Extract TOTAL ABS RETURNS
    # --------------------------------------------------------------
    pre_total  = np.abs(extract_abs(mats["NVDA"], pre_mask))
    post_total = np.abs(extract_abs(mats["NVDA"], post_mask))
    day0_total = np.abs(extract_abs(mats["NVDA"], day0_mask))
    observed_d0_total = float(np.nanmean(day0_total)) if len(day0_total) else np.nan

    # --------------------------------------------------------------
    # 3. Extract IDIOSYNCRATIC ABS RETURNS
    # --------------------------------------------------------------
    pre_idio  = np.abs(extract_abs(mats["Idio_Return"], pre_mask))
    post_idio = np.abs(extract_abs(mats["Idio_Return"], post_mask))
    day0_idio = np.abs(extract_abs(mats["Idio_Return"], day0_mask))
    observed_d0_idio = float(np.nanmean(day0_idio)) if len(day0_idio) else np.nan

    # --------------------------------------------------------------
    # Bootstrap helper
    # --------------------------------------------------------------
    def run_boot(pre_vec, post_vec, day0_obs):
        out = {}

        # ------------------ DAY-0 TEST ------------------
        if len(pre_vec) > 0:
            null_sample = pre_vec.flatten()
            (ci_d0, p_d0) = bootstrap_ci_pvalue_vol(
                null_sample=null_sample,
                observed_value=day0_obs,
                ci_level=ci_level,
                n_boot=n_boot
            )
        else:
            ci_d0 = (np.nan, np.nan)
            p_d0 = np.nan

        out["day0"] = {
            "observed": day0_obs,
            "ci": ci_d0,
            "p": p_d0
        }

        # ------------------ PRE–POST TEST ------------------
        if len(pre_vec) > 0 and len(post_vec) > 0:
            m = min(len(pre_vec), len(post_vec))
            pre_s  = pre_vec[:m]
            post_s = post_vec[:m]

            observed_diff = float(np.mean(post_s) - np.mean(pre_s))

            # Bootstrap difference distribution
            boot_vals = np.zeros(n_boot)
            for i in range(n_boot):
                b_pre  = np.random.choice(pre_s, size=m, replace=True)
                b_post = np.random.choice(post_s, size=m, replace=True)
                boot_vals[i] = np.mean(b_post) - np.mean(b_pre)

            ci_diff = (
                float(np.percentile(boot_vals, (1 - ci_level) * 50)),
                float(np.percentile(boot_vals, 100 - (1 - ci_level) * 50))
            )

            p_diff = float(
                np.mean(
                    np.abs(boot_vals - boot_vals.mean()) >=
                    np.abs(observed_diff - boot_vals.mean())
                )
            )
        else:
            observed_diff = np.nan
            ci_diff = (np.nan, np.nan)
            p_diff = np.nan

        out["diff"] = {
            "observed": observed_diff,
            "ci": ci_diff,
            "p": p_diff
        }

        return out

    # --------------------------------------------------------------
    # 4. Build outputs for Total and Idiosyncratic
    # --------------------------------------------------------------
    results_total = run_boot(pre_total, post_total, observed_d0_total)
    results_idio  = run_boot(pre_idio, post_idio, observed_d0_idio)

    return results_total, results_idio


def generate_hypothesis_conclusion(observed, ci, pvalue, metric_name):
    """
    Creates plain-language interpretation for statistical results.
    """
    ci_low, ci_high = ci

    # Base message
    msg = f"**{metric_name}**: "

    # Check CI exclusion
    if np.isnan(ci_low) or np.isnan(ci_high):
        return msg + "Insufficient data to form a conclusion."

    # Direction
    if observed > 0:
        direction = "increase"
    elif observed < 0:
        direction = "decrease"
    else:
        direction = "no change"

    strong_sig = pvalue < 0.005
    sig = pvalue < 0.05

    # CI significance logic
    ci_significant = not (ci_low <= 0 <= ci_high)

    # Combine logic
    if strong_sig and ci_significant:
        msg += (
            f"There is **very strong statistical evidence** (p = {pvalue:.4f}) "
            f"that {metric_name} experienced a **significant {direction}**. "
            f"Confidence interval [{ci_low:.4f}, {ci_high:.4f}] does not include 0."
        )
    elif sig and ci_significant:
        msg += (
            f"There is **strong evidence** (p = {pvalue:.4f}) supporting a "
            f"**meaningful {direction}** in {metric_name}. "
            f"Confidence interval [{ci_low:.4f}, {ci_high:.4f}] excludes 0."
        )
    elif sig and not ci_significant:
        msg += (
            f"Results suggest a **possible {direction}** (p = {pvalue:.4f}), "
            f"but the confidence interval [{ci_low:.4f}, {ci_high:.4f}] includes 0, "
            f"so the evidence is **inconclusive**."
        )
    else:
        msg += (
            f"No statistically significant change detected (p = {pvalue:.4f}). "
            f"Confidence interval [{ci_low:.4f}, {ci_high:.4f}] overlaps 0."
        )

    return msg

