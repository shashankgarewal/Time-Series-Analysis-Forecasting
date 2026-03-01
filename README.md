# Time Series Analysis & Forecasting

Exploring statistical approaches to volatility modeling and probabilistic forecasting on time-series data — from raw data inspection through to a regime-aware Monte Carlo simulation engine. This may further extend to ML-based approaches.

> For results walkthrough and plot analysis → [insights/](insights/)

---

## Notebooks

The research and experimentation trail behind the app. See [notebooks/](notebooks/) for the full breakdown.

| Notebook | What it covers |
|----------|----------------|
| `01_data_inspection` | Data quality checks, sanity checks, raw series inspection |
| `02_eda_analysis` | Full EDA — horizon selection rationale, volatility clustering, regime structure |
| `02_eda_analysis_limited_horizon` | Same EDA on ~1 year of data, showing why short horizons are insufficient |
| `03_statistical_model` | Regime detection and GARCH/EGARCH model fitting |
| `04_probabilistic_forecasting` | HMM regime detection, Monte Carlo simulation, price fan chart |

---

## App

**Live demo:** [regime-vol-forecaster.onrender.com](https://regime-vol-forecaster.onrender.com/)

A regime-switching volatility framework that segments return dynamics into LOW and HIGH states, fits regime-specific conditional variance models to each, and exposes the full pipeline — training, forecasting, simulation, and walk-forward evaluation — through a Flask API and browser UI.

1. **Detects** structural breaks using a hysteresis state machine or 2-state Gaussian HMM
2. **Fits** GARCH(1,1) on LOW-regime returns, EGARCH(1,1)+Student-*t* on HIGH-regime returns
3. **Forecasts** next-step conditional volatility given the current inferred regime
4. **Simulates** N Monte Carlo price paths with regime-conditioned distributional assumptions
5. **Evaluates** via walk-forward backtesting with periodic recalibration

No notebook required — everything runs through the API or the single-page UI.

---

## Regime Detectors

- **Hysteresis** — quantile thresholds with entry/exit hysteresis and a persistence filter. Interpretable, fast, no training instability.
- **HMM** — 2-state Gaussian HMM on `[log_return, rolling_vol]`. Data-driven state transitions, no manual threshold tuning.

## Volatility Models

- **LOW regime:** `GARCH(1,1)` with normal innovations
- **HIGH regime:** `EGARCH(1,1)` with Student-*t* innovations — captures leverage effects and fat tails

---

## Quickstart

Clone or fork the repo to run locally or build on top of it:

```bash
git clone https://github.com/shashankgarewal/time-series-analysis-forecasting
cd time-series-analysis-forecasting

pip install -r requirements.txt

python src/app.py
```

Navigate to `http://127.0.0.1:5000`.

To change the default ticker, date range, or model hyperparameters, edit `src/config.py`:

```python
TICKER       = "MSFT"
START_DATE   = "2014-01-01"
TRAIN_CUTOFF = "2022-01-01"
SCALE        = 100
```

---

## Key Design Decisions

**Why hysteresis instead of hard thresholds?**
A single spike above a quantile shouldn't flip regime state. The hysteresis detector requires the signal to cross a higher entry threshold and fall to a lower exit threshold before switching, plus a minimum persistence filter. This avoids noise-driven regime churn.

**Why EGARCH for the HIGH regime?**
EGARCH captures the leverage effect (negative return → disproportionate vol spike) and asymmetric shock response. Combined with Student-*t* innovations, it correctly fits the fat-tailed return distribution characteristic of stress periods.

**Why Student-*t* shocks in simulation?**
Drawing Gaussian shocks in Monte Carlo while the in-sample model is Student-*t* is a distributional inconsistency that underestimates tail risk. Shocks are drawn from the same fitted Student-*t* to keep simulation coherent with the model.

**Why walk-forward over a fixed test split?**
A single train/test split can coincidentally fall on an easy period. Walk-forward with periodic recalibration every `ROLLING_WINDOW` days mirrors live deployment — the model sees only past data at each step and must generalise across shifting regimes.

---

## Project Structure

```
├── README.md
├── requirements.txt
├── additional_info/        # Domain knowledge references and research notes
├── notebooks/              # Research trail — see notebooks/README.md
├── insights/               # Results walkthrough and plot analysis
├── src/
│   ├── app.py              # Flask API — routes, state management, serialisation
│   ├── config.py           # All hyperparameters in one place
│   ├── data.py             # yfinance download, log returns, rolling vol features
│   ├── regime.py           # HysteresisDetector and HMMDetector
│   ├── models.py           # RegimeVolModel — fit, forecast, evaluate, walk-forward
│   └── simulation.py       # Monte Carlo engine + fan chart percentiles
└── templates/
    └── index.html          # Single-page UI (Chart.js)
```

---


---

## Future Scope

- **Regime and model variants** — extend to a 3-state regime (LOW / MEDIUM / HIGH), swap in GJR-GARCH for asymmetric shock response, or fit a Student-*t* HMM for heavier-tailed state transitions; all are incremental changes within the existing architecture
- **Stateful regime memory** — currently each prediction is stateless beyond the rolling window; incorporating explicit regime duration and transition history as features could improve stability at regime boundaries
- **Online learning** — extend toward streaming parameter updates (recursive GARCH, online EM for the HMM) so the model adapts continuously as new data arrives, rather than requiring a full retrain
- **ML direction** — notebook 05 has the scaffolding in place: 5-day forward return target, lag/vol features, and HMM regime probabilities as soft inputs; next step is plugging in a classifier and evaluating directional accuracy

---

## Author

**Contact:** [shashankgarewal4+project@gmail.com](mailto:shashankgarewal4+project@gmail.com)

**Follow:** [linkedin.com/in/shashankgarewal](https://www.linkedin.com/in/shashankgarewal)

---

*Not financial advice.*