---

# NVIDIA Earnings Analysis
**Author:** Nawaz Pasha
**Tech Stack:** Python · Streamlit · Pandas · NumPy · Plotly · SciPy · scikit-learn (optional utilities)

---

## 📌 Overview

This repository contains a **fully modular Streamlit dashboard** that analyzes **NVIDIA (NVDA) stock behavior around earnings announcements** using:

* Event-study methodology
* Factor-based return decomposition
* Volatility analysis and regime detection
* Bootstrap-based statistical reliability testing

The dashboard is designed to answer **three core questions**:

1. **How does NVIDIA behave before, during, and after earnings?**
2. **How much of earnings-day movement is market-driven vs company-specific?**
3. **Are earnings-related returns and volatility statistically reliable or just noise?**

The result is a **research-grade yet executive-friendly** dashboard that blends quantitative rigor with intuitive explanations.

---

## 🧠 High-Level Methodology

### 1. Return Decomposition Framework

Each daily NVIDIA return is decomposed into:

* **Systematic (Factor-Predicted) Return**
  [
  \sum (\beta_i \times \text{Factor}_i)
  ]

* **Idiosyncratic Return**
  [
  \text{NVDA Return} - \text{Factor-Predicted Return}
  ]

This separation allows the dashboard to distinguish **market-driven moves** from **earnings-specific surprises**.

---

### 2. Earnings Event Alignment

* Earnings are released **after market close**
* **Day-0 = next trading day**
* Analysis uses a configurable **±N-day event window**

All statistics, charts, and tests are aligned to this event structure.

---

### 3. Volatility Measurement

The dashboard evaluates volatility using multiple lenses:

* Rolling volatility (annualized)
* Event-aligned absolute return changes
* Heatmaps across earnings cycles
* **EWMA volatility (λ = 0.94)** for regime detection

---

### 4. Statistical Reliability

Because earnings datasets are **small and heavy-tailed**, the dashboard relies on:

* Bootstrap confidence intervals
* Non-parametric testing
* Winsorization for return stability

This avoids misleading conclusions from standard parametric tests.

---

## 🧩 Code Architecture

The project is intentionally **modular and production-grade**.

```
.
├── app.py                     # Main Streamlit entry point
├── constants.py               # Colors, text blocks, explanations
├── data.py                    # Data loading & validation
├── analysis.py                # Core analytics & statistics engine
├── plots.py                   # Plotly visualization helpers
├── ui.py                      # Global CSS, sidebar branding, UI helpers
│
├── overview.py                # Tab 1: Executive overview & decomposition
├── event_analysis.py          # Tab 2: Earnings event return analysis
├── volatility_statistics.py   # Tab 3: Volatility & statistical reliability
├── summary.py                 # Tab 4: Narrative synthesis & conclusions
```

### Design Principles

* **Separation of concerns** (UI ≠ analytics ≠ plotting)
* **Pure functions** for analysis
* **Presentation-only tabs** (no data mutation)
* Reusable, testable components

---

## 📊 Dashboard Tabs Explained

---

## 🟦 Tab 1 — Overview

**Purpose:** Provide a **high-level executive snapshot** of the dataset and return structure.

### Components

* Dataset integrity KPIs (missing values, duplicates, outliers)
* Methodology explanation (plain language)
* Summary statistics table:

  * Mean, volatility, skew, kurtosis
  * Total vs systematic vs idiosyncratic returns
* Daily and cumulative return decomposition charts
* Factor model performance KPIs (R², correlation, hit rate)
* Top contributing factors
* Factor return correlation heatmap

**Key Insight:**

> Outside earnings, NVIDIA behaves like a factor-driven stock; on earnings day, idiosyncratic effects dominate.

---

## 🟩 Tab 2 — Earnings Event Analysis

**Purpose:** Analyze **return behavior around earnings** across multiple horizons.

### Components

* Event-aligned average return curves
* Cumulative event returns
* Pre / Event / Post comparisons
* Pattern tables across:

  * 1D, 2D, 1W, 2W windows
* Upside vs downside asymmetry metrics
* Risk ratios and dominance measures

**Key Insight:**

> Earnings Day (Day-0) is always the dominant move; pre- and post-drifts exist but are secondary.

---

## 🟥 Tab 3 — Earnings Volatility & Statistical Reliability

**Purpose:** Quantify **how volatility behaves** and test whether patterns are statistically meaningful.

### Volatility Analysis

* Rolling volatility with earnings markers
* Pre vs post absolute return changes
* Multi-window volatility comparison
* Event-level volatility tables

### Advanced Analytics

* Volatility heatmaps (raw & z-score)
* EWMA volatility curves
* Volatility burst classification
* Bootstrap significance testing:

  * Volatility
  * Returns
  * Directional bias

**Key Insight:**

> Volatility **always spikes** on earnings—even when returns are small.

---

## 🟨 Tab 4 — Summary & Key Findings

**Purpose:** Deliver a **decision-maker narrative** tying everything together.

### Contents

* Earnings performance synthesis
* Volatility regime interpretation
* Factor-model breakdown on Day-0
* Reliability vs randomness discussion
* Limitations and caveats
* Executive-level conclusions

**Key Insight:**

> Earnings reactions are patterned, regime-dependent, and volatility-driven—even when returns are statistically noisy.

---

## ⚠️ Limitations & Caveats

* Limited number of earnings events
* Heavy-tailed return distributions
* Regime shifts (e.g., AI cycle effects)
* Model results depend on factor selection
* Not financial advice — analytical framework only

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Ensure the following files are present:

* `01_case_study_returns.csv`
* `02_case_study_factor_loadings.csv`
* `03_case_study_earnings_dates.csv`

---

## 📬 Contact

**Nawaz Pasha**
📧 [Navvu18@gmail.com](mailto:Navvu18@gmail.com)
📞 +91-9986551895

---
