# Time Series Analysis & Forecasting Notebooks

The objective is to build an structured and evidence-driven workflow:

> Data Inspection → EDA & Statistical Diagnostics → Statistical Modeling → Probabilistic Forecasting → ML Extension (planned)


---

## Project Workflow

The project is organized into seperate notebooks, each focusing and serving a different purpose:

---

## [01_initial_data_inspection.ipynb](01_initial_data_inspection.ipynb)  
**Purpose:** Validate data integrity and establish baseline understanding.

Completed steps:
- Load MSFT stock data from Yahoo Finance database
- Validate dataset structure and date handling  
- Check missing values and missing trading days  
- Perform basic trend visualization (daily/weekly/monthly)  
- Compute returns and inspect extreme movement days  

**Outcome:**

- No structural data quality issues detected  
- Price levels exhibit non-stationarity  
- Extreme price movements align with high-volume event days  


---

### 2. [02_exploratory_data_analysis.ipynb](02_exploratory_data_analysis.ipynb) 
**Purpose:** Diagnose statistical properties relevant to forecasting.

Completed analysis:
- Horizon selection
- Return and log-return distribution inspection
- Classical and STL decomposition (multi-horizon)  
- Stationarity diagnostics  
- ACF / PACF Autocorrelation analysis on log-returns  
- Rolling standard deviation and volatility clustering check
- Calendar effect check

### Key Findings

- Price series is non-stationary in levels  
- Log-returns shows weak stationarity  
- Log-returns exhibit minimal linear autocorrelation  
- Decomposition may've display apparent seasonality driven by volatility structure rather than true periodic behavior  

**Conclusion:**  
Linear autoregressive structure alone is insufficient. Deliberate feature engineering is required before modeling.

> Also see [02_eda_analysis_limited_horizon.ipynb](02_eda_analysis_limited_horizon.ipynb) — same EDA performed on a ~1 year window, demonstrating why short horizons are insufficient for capturing volatility persistence and regime structure.

---

### [03_statistical_model.ipynb](03_statistical_model.ipynb)
**Purpose:** Establish a statistical baseline for volatility forecasting and introduce regime-aware modeling.

- ARCH-LM test confirms conditional heteroscedasticity (p << 0.05) — variance modeling is statistically justified
- Grid search over ARCH(p) and GARCH(p,q) for p,q ∈ {1,2,5,10,15} by AIC/BIC. Key finding: GARCH(1,1) = ARCH(∞) with geometrically decaying weights; high-lag variants add no information
- Excess kurtosis confirmed — GARCH with Student-t errors improves tail capture but lags 3–5 days, insufficient for rapid shifts
- Hysteresis regime detector: enter HIGH at q=0.80, exit at q=0.60, minimum 15-day persistence filter; isolated spikes (>3× rolling vol) treated separately
- GARCH(1,1) on LOW-regime returns, EGARCH(1,1) on HIGH-regime returns

**Conclusion:**  
Overall QLIKE improvement vs. single GARCH ~0.5% — modest on average, but ~25% lower variance MSE during HIGH-regime periods. Regime conditioning adds robustness where it matters, not uniformly.

---

### [04_probabilistic_forecasting.ipynb](04_probabilistic_forecasting.ipynb) 
**Purpose:** Replace hysteresis regime detection with a data-driven HMM and generate a probabilistic price forecast via Monte Carlo simulation.

- 2-state Gaussian HMM fitted on joint `[log_return, rolling_vol]` features — more stable boundaries than log-return alone
- HMM assigns ~30% of days to HIGH regime vs. ~5% from hysteresis — a fundamentally different definition, not a discrepancy
- GARCH(1,1) on LOW-regime returns, EGARCH(1,1) on HIGH-regime returns, fitted on regime-filtered training data
- Monte Carlo simulation: 1,000 paths over the full test window, regime-conditioned drift (10% annualised in LOW, 0% in HIGH), Gaussian shocks scaled by current conditional vol
- Fan chart output: p5/p25/p50/p75/p95 bands with actual MSFT price overlaid as out-of-sample reference

---

### [05_ml_model.ipynb](05_ml_model.ipynb) 🚧 (planned)
**Purpose:** Extend the pipeline with an ML-based return direction classifier on top of the HMM regime features.

Early scaffolding in place:
- 5-day forward cumulative log-return as the prediction target
- Binary direction label (`target = 1` if forward return > 0)
- Feature set started: lag-1 return, 5-day cumulative lag return, 10-day rolling vol, standardised shock (lag1 / vol_10), HMM regime state probabilities as soft features

Not actively being developed. The feature engineering foundation is there if picked up later.

---

## Dataset

Source: Yahoo Finance via `yfinance`

Available fields: Open, High, Low, Close, Volume

---

## Tools and Libraries

- Python, Pandas, NumPy
- Matplotlib, Seaborn  
- yfinance  
- statsmodels, scipy (stationarity tests, ARCH-LM)
- arch (GARCH, EGARCH model fitting)
- hmmlearn (Gaussian HMM)


---

## Current Status

✅ Notebook 01 — Data inspection complete  
✅ Notebook 02 — EDA complete  
✅ Notebook 03 — Statistical modeling and regime detection complete  
✅ Notebook 04 — HMM regime detection and Monte Carlo simulation complete  
🚧 Notebook 05 — ML direction classifier, early scaffolding only

---

## Author & Contacts

📩[shashankgarewal4+project@gmail.com](mailto:shashankgarewal4+project@gmail.com)

🔗[Linkedin](https://www.linkedin.com/in/shashankgarewal/)