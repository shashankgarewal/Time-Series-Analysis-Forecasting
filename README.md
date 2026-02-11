# Time Series Analysis & Forecasting

This project explores time series analysis and forcasting techniques on Microsoft (MSFT) stock price data from 01-01-2025.

The objective is to build an structured and evidence-driven workflow:

> Data Inspection → Statistical Diagnostics → Feature Engineering → Modeling → Evaluation


---

## Project Workflow

The project is organized into seperate notebooks, each focusing and serving a different purpose:

---

## (01_initial_data_inspection.ipynb)[] ✅  
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

### 2. [02_exploratory_data_analysis.ipynb](02_exploratory_data_analysis.ipynb) ✅ 
**Purpose:** Diagnose statistical properties relevant to forecasting.

Completed analysis:
- Horizon selection
- Return and log-return distribution inspection
- Classical and STL decomposition (multi-horizon)  
- Stationarity diagnostics  
- ACF / PACF Autocorrelation analysis on log-returns  
- Rolling standard deviation and volatility clustering check

### Key Findings

- Price series is non-stationary in levels  
- Log-returns shows weak stationarity  
- Log-returns exhibit minimal linear autocorrelation  
- Decomposition may've display apparent seasonality driven by volatility structure rather than true periodic behavior  

**Conclusion:**  
Linear autoregressive structure alone is insufficient. Deliberate feature engineering is required before modeling.

---

### 3. [03_feature_engineering.ipynb](03_feature_engineering.ipynb) *(In Progress)*
**Purpose:** Construct model-ready predictive features.

Planned steps:
- Define forecasting target (next-day log-return or direction)  
- Create multi-lag return features  
- Engineer rolling statistics (volatility, momentum)  
- Develop regime-based indicators  
- Implement time-aware train/test splits  
- Ensure no look-ahead bias  
- Export modeling-ready dataset  

---

### 4. Forecasting and Modeling *(Planned)*
In this stage, different forecasting approaches will be explored based on the insights from EDA.  
Candidate methods may include:

- Simple baseline models  
- Classical statistical time series models  
- Machine learning-based regression approaches  
- Deep learning sequence models (if appropriate)

Final model choice will depend on data behavior and evaluation results.

➡️ Notebook: [04_modeling.ipynb](04_modeling.ipynb)

---

### 5. Evaluation *(Planned)*
Model performance will be evaluated using time-series appropriate validation strategies such as walk-forward testing and forecasting error metrics.

➡️ Notebook: [05_evaluation.ipynb](05_evaluation.ipynb)

---

## Dataset

Source: Yahoo Finance via `yfinance`

Available fields: Open, High, Low, Close, Volume

---

## Tools and Libraries

- Python, Pandas, NumPy
- Matplotlib, Seaborn  
- yfinance  
- statsmodels (for time series diagnostics)

---

## Current Status

✅ Notebook 01 completed  
✅ Notebook 02 completed
🚧 Notebook 03 currently in progress
📌 Modeling stage will be finalized during feature engineering

---

## Next Steps

The immediate next step is to perform deeper exploratory analysis on return behavior, volatility patterns, and stationarity before selecting appropriate forecasting models.

## Author & Contacts
📩[shashankgarewal4+project@gmail.com](mailto:shashankgarewal4+project@gmail.com)
🔗[Linkedin](https://www.linkedin.com/in/shashankgarewal/)