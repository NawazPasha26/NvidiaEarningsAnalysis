"""
@author - Nawaz Pasha
Constants update
"""

# constants.py
#This module will have all the constants declared, like texts, messages etc that will be used for display purpose on the dashboard
#This will be called and used in the main moduke which is app.py

COLORS = {
    "blue": "#1f77b4", "orange": "#ff7f0e", "green": "#2ca02c", "red": "#d62728",
    "purple": "#9467bd", "brown": "#8c564b", "pink": "#e377c2", "gray": "#7f7f7f",
    "olive": "#bcbd22", "teal": "#17becf"
}
PALETTE = [COLORS[c] for c in ["orange", "green", "red", "purple", "brown", "pink", "gray", "olive", "teal"]]

TXT_EARNINGS_NOTE = (
    "Earnings releases occur after market close; price reaction is on the next trading day (day 0)."
)
TXT_VOL_EXPLAIN = """\
## Understanding Volatility Around Earnings  
- **What is Volatility?** Volatility measures how much a stock's price moves up and down. High = big swings; Low = steadier moves.  
- **Rolling Volatility** smooths noise by measuring volatility over a moving window (e.g., 20 days).  
- **Annualized Volatility** scales daily volatility to a yearly number (e.g., "~30% annualized").  
- **Pre/Event day/Post**: days after earnings; first trading day after results; days after earnings. 
- **Why it matters**: Volatility often **spikes** on results and **settles** as uncertainty clears.
"""
TXT_DATA_EDA = """\
### What this data contains  
- **NVDA daily returns**: NVIDIA's stock return for each trading day.  
- **Factor returns**: Daily returns for different factors (e.g., Market, Momentum, Quality, Semiconductors).  
- **Daily factor loadings (betas)**: NVIDIA's day-by-day exposure to each factor.  

### How we use it  
- We compute the **factor-predicted return** as the sum of **(beta × factor return)** across selected factors.  
- The **idiosyncratic return** is the **residual** (actual NVIDIA return minus factor-predicted); this captures company-specific surprise.  
- We check **data quality** (missing values, duplicates, outliers) so event-study results are not driven by bad prints or fat fingers.  

### Why the correlation heatmap matters  
- It shows how selected **factor returns co-move**. High correlations mean those factors often move together; low/negative correlations can help diversify factor exposure.
"""
TXT_LIMITS = """\
- **Few events**: Fewer earnings dates -> weaker statistical power and wider uncertainty.  
- **Model coverage**: If important risk drivers are missing or betas are noisy, the **idiosyncratic** piece may be overstated.  
- **After-hours timing**: Earnings happen after market close; if reaction to **day 0** can mix pre/post market moves.
- **Non-normal returns**: Stock returns can be jumpy; t-tests are **approximations**. Use results as a guide, not a guarantee.  
- **Parameter sensitivity**: Changing the **factors** used or the **rolling-window** length will change results.
"""
