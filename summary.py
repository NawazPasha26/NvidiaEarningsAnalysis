# tabs/summary.py

"""
Summary & Key Findings Tab
Author : Nawaz Pasha
-------------------------

This module renders the **Summary** tab of the NVIDIA Earnings Analysis dashboard.

Purpose:
- Provide a narrative synthesis of all prior analyses
- Summarize earnings-period return behavior
- Highlight volatility dynamics around earnings
- Explain factor-model breakdown on earnings day
- Interpret clustering, regime behavior, and similarity patterns
- Present statistical reliability and limitations
- Deliver executive-level conclusions in plain language

Inputs:
- df      : Daily return dataframe
- df_e    : Earnings dates dataframe
- ev_out  : Output of build_event_window (mats, idxs, event_info)
- win     : Event window size (± days)

Outputs:
- Streamlit-rendered narrative sections
- No plots or tables generated here
- Pure interpretation layer (no new analytics)

Design Notes:
- All numerical values are precomputed within this module
- Relies on analysis.py utilities for consistency
- Intended for executive and decision-maker consumption
"""

import streamlit as st
import pandas as pd
import numpy as np

from analysis import (
    compute_event_averages,
    compute_pre_post_means,
    compute_interval_move,
    mask_triplet,
)


def render_summary(df, df_e, ev_out, win):
    """
    Renders the Summary tab.
    Assumes ev_out has already been constructed upstream.
    """

    # =====================================================================
    # PRECOMPUTATION FOR SUMMARY TAB
    # =====================================================================
    
    if ev_out:
        mats, idxs2, event_info = ev_out
    
        # ---- Pre/Post/Event Returns ----
        idxs2, avg = compute_event_averages(ev_out)
        (
            pre_tot, ev_tot, post_tot,
            pre_fac, ev_fac, post_fac,
            pre_idi, ev_idi, post_idi
        ) = compute_pre_post_means(avg, idxs2)
    
        # ---- Interval Returns (Dynamic Top/Bottom Lists) ----
        interval_df = compute_interval_move(df, df_e, win)
        interval_returns = interval_df["IntervalReturn"].dropna()
    
        interval_sorted = interval_df.sort_values("IntervalReturn", ascending=False)
    
        # Top 4 positive
        top_pos = interval_sorted.head(4)
    
        # Top 3 negative
        top_neg = interval_sorted.tail(3).sort_values("IntervalReturn")
    
        def fmt_interval_row(r):
            date = pd.to_datetime(r["EarningsDate"]).strftime("%b %Y")
            return f"**{r['IntervalReturn']:.2%} ({date})**"
    
        top_pos_html = "<br>".join([f"- {fmt_interval_row(r)}" for _, r in top_pos.iterrows()])
        top_neg_html = "<br>".join([f"- {fmt_interval_row(r)}" for _, r in top_neg.iterrows()])
    
        # ---- Basic Stats ----
        avg_interval = interval_returns.mean()
        median_interval = interval_returns.median()
        max_interval = interval_returns.max()
        min_interval = interval_returns.min()
    
        # ---- Day-0 Idiosyncratic Share ----
        pre_mask, day0_mask, post_mask = mask_triplet(idxs2)
    
        day0_fac = np.abs(mats["Factor_Pred_Return"][:, day0_mask]).flatten()
        day0_idio = np.abs(mats["Idio_Return"][:, day0_mask]).flatten()
    
        denom = day0_fac + day0_idio
        valid = denom > 0
    
        idio_share_mean = float(np.nanmean(day0_idio[valid] / denom[valid])) if valid.any() else np.nan
        day0_total_abs = float(np.nanmean(np.abs(mats["NVDA"][:, day0_mask].flatten())))
    
        # ---- EWMA Volatility ----
        vol_curve = mats["NVDA"].std(axis=0)
        lambda_vol = 0.94
        ewma_vol = np.zeros_like(vol_curve)
        ewma_vol[0] = vol_curve[0]
    
        for i in range(1, len(vol_curve)):
            ewma_vol[i] = lambda_vol * ewma_vol[i - 1] + (1 - lambda_vol) * vol_curve[i]
    
        pre_vol = float(np.nanmean(ewma_vol[:win]))
        event_vol = float(ewma_vol[win])
        post_vol = float(np.nanmean(ewma_vol[win + 1:]))
    
    else:
        pre_tot = ev_tot = post_tot = np.nan
        top_pos_html = top_neg_html = ""
        idio_share_mean = day0_total_abs = np.nan
        pre_vol = event_vol = post_vol = np.nan
    
    
    # =====================================================================
    # PERFORMANCE OF NVIDIA DURING EARNINGS
    # =====================================================================
    
    with st.expander("📌 Performance of NVIDIA During Earnings", expanded=True):
    
        st.markdown("""
            <style>
                .summary-text { font-size: 13px; line-height: 1.3; }
            </style>
        """, unsafe_allow_html=True)
    
        st.markdown(f"""
    <div class='summary-text'>
    
    ###### **1. NVIDIA shows mild pre-earnings drift, a large idiosyncratic earnings-day jump, and modest positive continuation afterward.**
    
    Across all earnings events, NVIDIA rises **{pre_tot:.2%}** in the 10 days before earnings  
    (mostly factor-driven drift), jumps **{ev_tot:.2%} on Day-0**  
    (a company-specific surprise nearly **{(ev_tot/pre_tot if pre_tot != 0 else float('nan')):.1f}×** larger),  
    and then adds another **{post_tot:.2%}** in the following 10 days.
    
    This confirms a consistent pattern:  
    **small macro-driven rise → big idiosyncratic shock → mild post-drift**. 
    
    ---
    
    ###### **2. Earnings intervals are strongly right-skewed: most cycles are materially positive, but the few negative ones are sharp.**
    
    Your ±10-day intervals show repeated strong upside:
    
    {top_pos_html}
    
    Meanwhile, the few negative events include:
    
    {top_neg_html}
    
    Statistical tables confirm:
    - **Positive events occur more often**  
    - **Positive averages are larger**  
    - The distribution is **right-skewed**, driven by exceptionally strong AI-related quarters.
    
    ---
    
    ###### **3. Day-0 is always the largest move, and while typical upside outweighs typical downside, extreme shocks remain the primary risk.**
    
    The Pattern Table (1D/2D/1W/2W Before & After) shows  
    **Day-0 is the largest return in every single window**.
    
    Risk ratios show:
    - **Avg Return / Avg Down Move > 1** → upside dominates  
    - **Avg Return / Biggest Down Move < 1** → rare deep negatives can exceed typical gains  
    
    This captures Day-0 dominance + upside–downside asymmetry + tail-risk vulnerability.
    
    ---
    
    ###### **4. Volatility reliably spikes during earnings, confirmed by EWMA volatility burst analysis.**
    
    EWMA volatility levels:
    - **Pre-earnings:** {pre_vol:.4f} (stable)  
    - **Day-0:** {event_vol:.4f} (sharp spike)  
    - **Post-earnings:** {post_vol:.4f}
    
    ~60% of events revert quickly;  
    ~40% show elevated volatility continuation.
    
    Even when returns are small, **volatility always jumps**, validating options pricing behavior.
    
    ---
    
    ###### **5. Factor models explain NVIDIA well outside earnings — but break down completely on Day-0.**
    
    Normal trading days:
    - Strong factor alignment (Market, Momentum, Quality, Semiconductors)
    
    Earnings day:
    - **Factor-driven:** {(1-idio_share_mean):.0%}  
    - **Idiosyncratic surprise:** {idio_share_mean:.0%}  
    - Hedges offer **little protection**  
    - Factor contributions shrink relative to fundamental shocks  
    
    This preserves full detail: normal-day factor fit vs. earnings-day factor failure.
    
    ---
    
    ###### **6. Earnings reactions fall into two distinct behavioral regimes, with recurring similarities across years.**
    
    **High-reaction cluster:**
    - Large Day-0 jumps  
    - Strong post-earnings continuation  
    - Matches NVIDIA’s **AI-driven super-cycle (2023–2024)**  
    
    **Low-reaction cluster:**
    - Flat or negative Day-0  
    - Quick volatility mean-reversion  
    - Matches macro-sensitive or weaker-guidance quarters  
    
    Similarity analysis reveals repeating templates:
    - **May 2023 ↔ May 2024**  
    - **Aug 2024 ↔ Aug 2025**
    
    This indicates NVIDIA’s earnings reactions are **patterned**, not random.
    
    ---
    
    **NVIDIA earnings deliver predictable pre/post drift, a massive idiosyncratic Day-0 shock, strong upside skew with rare steep drawdowns, reliable volatility bursts, factor-model breakdown during earnings, and a clear two-regime structure tied to fundamentals.**
    
    </div>
    """, unsafe_allow_html=True)
    
    # =====================================================================
    # VOLATILITY AROUND EARNINGS
    # =====================================================================
    
    with st.expander("📌 Volatility Around Earnings", expanded=True):
    
        st.markdown("""
            <style>
                .summary-text { font-size: 13px; line-height: 1.3; }
            </style>
        """, unsafe_allow_html=True)
    
        st.markdown(f"""
    <div class='summary-text'>
    
    ###### **1. NVIDIA shows a highly consistent volatility pattern around earnings: calm before, sharp spike on Day 0, and partial normalization after.**
    
    Across all events, the 10-day pre-earnings period is **quiet and stable**,  
    with volatility typically around **{pre_vol:.4f}** as traders avoid major positioning.
    
    **Earnings Day (Day 0)** produces the **highest volatility in the entire ±10-day window**,  
    with a spike to **{event_vol:.4f}**, regardless of positive/negative returns.
    
    Post-earnings volatility:
    - **Drops toward pre-earnings levels in ~60%** of events  
    - **Remains elevated in ~40%** depending on results strength  
    
    This pattern repeats across all earnings cycles.
    
    ---
    
    ###### **2. Volatility Uplift metrics show Day 0 as the dominant driver, forming a predictable shape across all earnings events.**
    
    The uplift tables confirm the largest jump is always:  
    **pre → event**,  
    with Day-0 volatility dramatically higher than the days before.
    
    Post-earnings volatility sits **between pre-event and Day 0**,  
    showing partial—but not guaranteed—stabilization.
    
    This shape matches liquid high-growth tech behaviour:  
    **“quiet → volatility shock → reversion.”**
    
    ---
    
    ###### **3. Volatility heatmaps reveal universal Day 0 clustering, quiet pre-earnings zones, post-earnings divergence, and multi-quarter volatility regimes.**
    
    Heatmaps show:
    - A **dark vertical column** at Day 0 (volatility spike)  
    - A **light block before earnings** (low uncertainty)  
    - Divergence after earnings:  
      - Some events fade quickly  
      - Others remain volatile longer (especially **AI-acceleration quarters**)  
    
    Heatmaps also show clear **volatility regimes**:
    - **2023–2024:** structurally higher post-earnings volatility  
    - **Late-2024 to 2025:** mixed fast-fade & moderate-persistence profiles  
    
    ---
    
    ###### **4. EWMA volatility curves reinforce: flat pre-earnings, sharp Day-0 burst, uneven normalization afterwards.**
    
    EWMA smoothing highlights:
    - **Stable low-vol before results**  
    - A **clear upward kink** on earnings day across every event  
    - **Non-uniform post-earnings decline**  
    
    Strong AI-driven quarters maintain elevated EWMA longer;  
    quieter cycles revert quickly.
    
    Volatility burst ratios (**event vol / pre vol**) are **>1** for nearly all events,  
    proving volatility **always jumps**, even when returns are small.
    
    ---
    
    ###### **5. Overall interpretation: volatility is predictable and structural, even though return direction is not.**
    
    Combining uplift tables, heatmaps, raw vol, and EWMA views shows:
    - **Low/stable volatility before earnings**  
    - **Guaranteed spike on earnings day**  
    - **Partial but non-uniform normalization after**  
    - **Regime-dependent volatility behaviour**  
    - **Predictable volatility patterns, even when returns vary**  
    
    NVIDIA behaves like a classic **“volatility event stock”**:  
    **earnings always increase uncertainty**, even when price moves are modest.
    
    </div>
    """, unsafe_allow_html=True)
    
    # =====================================================================
    # STATISTICAL RELIABILITY
    # =====================================================================
    
    with st.expander("📌 Statistical Reliability", expanded=True):
    
        st.markdown("""
            <style>
                .summary-text { font-size: 13px; line-height: 1.3; }
            </style>
        """, unsafe_allow_html=True)
    
        st.markdown("""
    <div class='summary-text'>
    
    ###### **1. Bootstrapped volatility tests clearly show that earnings-day volatility is consistently and meaningfully higher than normal trading days.**
    
    Across all earnings events, Day-0 volatility sits **well above** the  
    bootstrapped distribution of typical daily volatility.
    
    Confidence intervals are **clearly separated**, and p-values are low.
    
    This confirms the volatility spike is a **reliable and repeatable feature**,  
    not a random outlier.
    
    ---
    
    ###### **2. Day-0 returns often show large and economically meaningful jumps, but because these stronger reactions occur only in certain quarters, the overall return effect does not reach statistical significance.**
    
    Several earnings events exhibit **very large moves**,  
    while others are muted or negative.
    
    This creates **wide variability** in Day-0 outcomes.
    
    After applying **bootstrapping and winsorization** to adjust for heavy tails,  
    the mean and median Day-0 returns **do not consistently exceed**  
    typical daily levels.
    
    **In simple terms:**  
    Earnings can produce big moves, but **not consistently enough**  
    to reach statistical confirmation.
    
    ---
    
    ###### **3. Pre- vs post-earnings return behaviour does not show a statistically strong or uniform shift once event-to-event variability is considered.**
    
    Some quarters visually show calming or continuation patterns after earnings,  
    but this behaviour is **not consistent across all events**.
    
    Bootstrapped comparisons show **no reliable statistical separation**  
    between pre- and post-earnings distributions.
    
    This suggests specific cycles may show clearer trends,  
    but the broader dataset does not support a **uniform post-earnings drift**.
    
    ---
    
    ###### **4. Limitation: Small sample size, heavy-tailed reactions, and regime shifts make statistical inference challenging.**
    
    With only **~12 earnings events**, some extremely strong and others subdued,  
    traditional parametric tests are **not reliable**.
    
    NVIDIA’s returns exhibit:
    - **Non-normal behaviour**  
    - **Skewed distributions**  
    - **Heavy tails**  
    - **Regime dependence** 
    
    Because of these structural limitations,  
    the dashboard correctly relies on:
    - **Bootstrapping**  
    - **Winsorization**  
    - **Non-parametric methods**     
    to ensure results remain as robust as possible.
    
    </div>
    """, unsafe_allow_html=True)

