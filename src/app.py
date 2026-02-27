# api.py

import math
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, render_template

import config
import data as data_module
from regime import HysteresisDetector, HMMDetector
from models import RegimeVolModel
from simulation import simulate, fan_chart

app = Flask(__name__)

state = {}


def trained():
    return state.get("ready", False)


def sanitize(obj):
    """Recursively replace NaN/Inf with None so jsonify produces valid JSON."""
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj

@app.errorhandler(Exception)
def handle_exception(e):
    app.logger.exception(e)
    return jsonify({"error": str(e)}), 500


@app.route("/")
def home():
    return render_template("index.html", ticker=config.TICKER, start=config.START_DATE, 
                           cutoff=config.TRAIN_CUTOFF, trained=trained())


@app.route("/train", methods=["POST"])
def train():
    body          = request.get_json(silent=True) or {}
    detector_name = body.get("regime_detector", "hysteresis")
    ticker        = body.get("ticker", "").strip().upper() or config.TICKER
    start         = body.get("start_date", config.START_DATE) or config.START_DATE
    cutoff        = body.get("train_cutoff", config.TRAIN_CUTOFF) or config.TRAIN_CUTOFF

    try:
        df, df_train, df_test = data_module.load(ticker, start, cutoff)
    except Exception:
        ticker = config.TICKER
        df, df_train, df_test = data_module.load(ticker, start, cutoff)

    features_train = data_module.build_features(df_train)
    features_test  = data_module.build_features(df_test)
    features_full  = data_module.build_features(df)

    if detector_name == "hmm":
        det           = HMMDetector().fit(features_train)
        high_train, _ = det.predict(features_train)
        high_test,  _ = det.predict(features_test)
        high_full,  _ = det.predict(features_full)
    else:
        det           = HysteresisDetector().fit(features_train["rolling_vol"])
        high_train, _ = det.predict(features_train["rolling_vol"])
        high_test,  _ = det.predict(features_test["rolling_vol"])
        high_full,  _ = det.predict(features_full["rolling_vol"])

    try:
        model = RegimeVolModel().fit(df_train["pctLogReturn"].dropna(), high_train)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    state.update({
        "ready":      True,
        "ticker":     ticker,
        "detector":   detector_name,
        "det":        det,
        "model":      model,
        "df":         df,
        "df_train":   df_train,
        "df_test":    df_test,
        "high":       high_train,
        "high_test":  high_test,
        "high_full":  high_full,
        "train_rows": len(df_train),
        "test_rows":  len(df_test),
    })
    
    rv      = features_train["rolling_vol"]
    regimes = high_train.reindex(rv.index).fillna(False)   # align, no NaN gaps

    scatter_dates   = [str(d.date()) for d in rv.index]
    scatter_vols    = [round(float(v), 6) for v in rv]
    scatter_regimes = ["high" if bool(r) else "low" for r in regimes]

    return jsonify(sanitize({
        "status":          "trained",
        "ticker":          ticker,
        "detector":        detector_name,
        "train_rows":      len(df_train),
        "test_rows":       len(df_test),
        "high_regime_pct": round(float(high_train.mean() * 100), 2),
        "garch_aic":       round(float(model.garch_fit.aic), 2),
        "egarch_aic":      round(float(model.egarch_fit.aic), 2),
        "scatter_dates":   scatter_dates,
        "scatter_vols":    scatter_vols,
        "scatter_regimes": scatter_regimes,
        "last_close":      round(float(df_train["Close"].iloc[-1]), 2),
        "test_size":       len(df_test),
    }))


@app.route("/forecast", methods=["POST"])
def forecast():
    if not trained():
        return jsonify({"error": "not trained"}), 503

    body  = request.get_json(silent=True) or {}
    model = state["model"]
    pct   = state["df"]["pctLogReturn"].dropna()
    high  = state["high_full"]

    if "regime" in body:
        regime = body["regime"]
    else:
        idx    = pct.index.intersection(high.index)
        regime = "high" if bool(high.loc[idx].iloc[-1]) else "low"

    pred_scaled = model.forecast_next(pct, regime)
    pred_raw    = pred_scaled / config.SCALE
    pred_annual = pred_raw * np.sqrt(252)

    pct_test  = state["df_test"]["pctLogReturn"].dropna()
    high_test = state["high_test"]
    eval_data = model.evaluate_on_test(pct_test, high_test)

    return jsonify(sanitize({
        "regime":         regime,
        "vol_scaled":     round(pred_scaled, 6),
        "vol_raw":        round(pred_raw, 6),
        "vol_annualised": round(pred_annual, 6),
        "eval":           eval_data,
    }))


@app.route("/simulate", methods=["POST"])
def simulate_route():
    if not trained():
        return jsonify({"error": "not trained"}), 503

    body     = request.get_json(silent=True) or {}
    df_test  = state["df_test"]
    model    = state["model"]
    det      = state["det"]
    pct      = state["df"]["pctLogReturn"].dropna()
    detector = state["detector"]

    last_price = float(body.get("last_price", float(state["df_train"]["Close"].iloc[-1])))
    horizon    = int(body.get("horizon", len(df_test)))
    n_paths    = int(body.get("n_paths", config.N_SIM))

    features_test = data_module.build_features(df_test)
    if detector == "hmm":
        high_test, _ = det.predict(features_test)
    else:
        # FIX: consistent with /train — predict on rolling_vol, not LogReturn
        rv_test      = features_test["rolling_vol"]
        high_test, _ = det.predict(rv_test)

    labels = pd.Series(
        ["high" if h else "low" for h in high_test.values],
        index=high_test.index,
    ).iloc[:horizon]

    if len(labels) < horizon:
        pad    = pd.Series([labels.iloc[-1]] * (horizon - len(labels)))
        labels = pd.concat([labels, pad], ignore_index=True)

    cond_vols = np.array([model.forecast_next(pct, r) for r in labels])

    # simulate() so HIGH-regime shocks are drawn from Student-t, not Normal.
    nu    = model.egarch_nu
    paths = simulate(
        labels.values, cond_vols, last_price,
        horizon=horizon, n_paths=n_paths, nu=nu,
    )
    fan = fan_chart(paths)

    # date labels for x-axis: test index up to horizon+1 (includes t=0)
    test_dates    = df_test.index[:horizon + 1]
    date_labels   = [str(d.date()) for d in test_dates]

    # actual closing prices over the same window (for overlay)
    actual_prices = [round(float(v), 2) for v in df_test["Close"].iloc[:horizon + 1]]

    return jsonify({
        "ticker":        state["ticker"],
        "last_price":    last_price,
        "horizon":       horizon,
        "n_paths":       n_paths,
        "median_final":  round(float(np.median(paths[:, -1])), 2),
        "min_final":     round(float(paths[:, -1].min()), 2),
        "max_final":     round(float(paths[:, -1].max()), 2),
        "fan_chart":     {col: fan[col].tolist() for col in fan.columns},
        "date_labels":   date_labels,
        "actual_prices": actual_prices,
    })


@app.route("/metrics", methods=["POST"])
def metrics():
    if not trained():
        return jsonify({"error": "not trained"}), 503

    body     = request.get_json(silent=True) or {}
    duration = int(body.get("duration", 60))

    model = state["model"]
    pct   = state["df"]["pctLogReturn"].dropna()
    high  = state["high_full"]

    result = model.walk_forward_eval(pct, high, duration)
    return jsonify(sanitize(result))


if __name__ == "__main__":
    app.run(host=config.HOST, port=config.PORT, debug=True)