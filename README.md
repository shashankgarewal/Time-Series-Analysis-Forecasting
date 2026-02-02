# MSFT Stock Time Series Analysis & Forecasting

This project explores Microsoft (MSFT) stock price data from 01-01-2025 using time series analysis techniques.  
The goal is to build a structured workflow starting from data inspection, followed by exploratory analysis, feature engineering, and forecasting experiments.

---

## Project Workflow

The project is organized into multiple notebooks, each focusing on a specific stage:

---

### 1. Initial Data Inspection ✅
- Load MSFT stock data from Yahoo Finance  
- Validate dataset structure and date handling  
- Check missing values and missing trading days  
- Perform basic trend visualization (daily/weekly/monthly)  
- Compute returns and inspect extreme movement days  

➡️ Notebook: [01_initial_data_inspection.ipynb](01_initial_data_inspection.ipynb)

---

### 2. Exploratory Data Analysis (EDA) *(In Progress)*
Planned analysis includes:
- Return distribution behavior  
- Rolling volatility and trend smoothing  
- Stationarity and autocorrelation diagnostics  

➡️ Notebook: [02_exploratory_data_analysis.ipynb](02_exploratory_data_analysis.ipynb)

---

### 3. Feature Engineering *(Planned)*
Potential features to explore:
- Lag-based predictors  
- Rolling statistics (moving averages, volatility)  
- Momentum-style indicators  
- Forecasting target definition  

➡️ Notebook: [03_feature_engineering.ipynb](03_feature_engineering.ipynb)

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

Stock price data is sourced from Yahoo Finance using the  library.  
Available fields include Open, High, Low, Close, Adjusted Close, and Volume.

---

## Tools and Libraries

- Python, Pandas, NumPy
- Matplotlib, Seaborn  
- yfinance  
- statsmodels (for time series diagnostics)

---

## Status

✅ Notebook 01 completed  
🚧 Notebook 02 currently in progress  
📌 Modeling stage will be decided after EDA findings

---

## Next Steps

The immediate next step is to perform deeper exploratory analysis on return behavior, volatility patterns, and stationarity before selecting appropriate forecasting models.

