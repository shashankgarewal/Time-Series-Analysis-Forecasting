# models.py
# regime-conditional vol models: GARCH(1,1) for LOW, EGARCH(1,1)+t for HIGH

import numpy as np
import pandas as pd
from arch import arch_model

import config

MIN_OBS = 30   # minimum observations needed to fit a model


class RegimeVolModel:

    # ---- small helpers ----
    def _am(self, r: pd.Series, regime: str):
        if regime == "high":
            return arch_model(r, mean="Zero", vol="EGARCH", p=config.P, q=config.Q, dist=config.HIGH_DIST)
        return arch_model(r, mean="Zero", vol="GARCH",  p=config.P, q=config.Q, dist="normal")

    # rolling variance for k-window 
    def _rv(self, r: pd.Series, k: int) -> pd.Series:
        return ((r ** 2).rolling(k).mean() ** 0.5).dropna()

    # ---- train ----
    def fit(self, pct_returns: pd.Series, high_mask: pd.Series) -> "RegimeVolModel":
        idx = pct_returns.index.intersection(high_mask.index)
        r   = pct_returns.loc[idx]
        h   = high_mask.loc[idx].astype(bool)

        low_returns  = r[~h]
        high_returns = r[h]

        if len(low_returns) < MIN_OBS:
            raise ValueError(f"Not enough LOW-regime samples to fit GARCH ({len(low_returns)} < {MIN_OBS})")
        if len(high_returns) < MIN_OBS:
            raise ValueError(
                f"Not enough HIGH-regime samples to fit EGARCH ({len(high_returns)} < {MIN_OBS}). "
            )

        self.garch_fit  = self._am(low_returns,  "low").fit(disp="off")
        self.egarch_fit = self._am(high_returns, "high").fit(disp="off")
        return self

    @property
    def egarch_nu(self) -> float:
        """Degrees-of-freedom from fitted Student-t EGARCH."""
        p = self.egarch_fit.params
        return float(p.get("nu", p.get("Nu", 8.0)))

    # ---- single-step forecast (fixed trained params) ----
    def forecast_next(self, pct_returns: pd.Series, regime: str) -> float:
        params  = self.egarch_fit.params if regime == "high" else self.garch_fit.params
        fixed   = self._am(pct_returns, regime).fix(params)
        predvar = float(fixed.forecast(horizon=1, method=("simulation" if regime == "high" else "analytic")).variance.iloc[-1, 0])
        return float(np.sqrt(max(predvar, 0.0)))  # guard tiny negatives

    # ---- eval 1: fixed-params on a contiguous test series ----
    def evaluate_on_test(self, pct_test: pd.Series, high_test: pd.Series) -> dict:
        idx         = pct_test.index.intersection(high_test.index)
        pct_returns = pct_test.loc[idx]
        high        = high_test.loc[idx].astype(bool)

        # both run on full contiguous returns; select by regime mask
        g_vol = self._am(pct_returns, "low").fix(self.garch_fit.params).conditional_volatility
        e_vol = self._am(pct_returns, "high").fix(self.egarch_fit.params).conditional_volatility

        pred_vol = pd.Series([e_vol.loc[d] if high.loc[d] else g_vol.loc[d] for d in pct_returns.index], 
                             index=pct_returns.index)

        rv  = self._rv(pct_returns, config.ROLLING_WINDOW)
        idx = pred_vol.index.intersection(rv.index)
        pred_vol, rv = pred_vol.loc[idx], rv.loc[idx]

        mae  = float((pred_vol - rv).abs().mean())
        rmse = float(((pred_vol - rv) ** 2).mean() ** 0.5)

        return {
            "dates":      [str(d.date()) for d in pred_vol.index],
            "pred_vol":   [round(float(v), 4) for v in pred_vol.tolist()],
            "actual_vol": [round(float(v), 4) for v in rv.tolist()],
            "mae":        round(mae, 4),
            "rmse":       round(rmse, 4),
        }

    # ---- eval 2: walk-forward (daily), refit params every k_refit ----

    def walk_forward_eval(self, pct_returns: pd.Series, high_mask: pd.Series, duration: int) -> dict:
        """
        Walk-forward back-test over last N days:
        - refit params every k_refit days
        - forecast daily using fixed params + expanding history
        """

        idx     = pct_returns.index.intersection(high_mask.index)
        returns = pct_returns.loc[idx]
        high    = high_mask.loc[idx].astype(bool)

        if duration <= 0:
            raise ValueError("duration must be > 0")
        if duration >= len(returns):
            raise ValueError("duration too large")


        rv = self._rv(returns, config.ROLLING_WINDOW)

        start = len(returns) - duration
        preds, dates = [], []

        params_low, params_high = None, None

        for step in range(duration):
            t = start + step
            if t <= 0:
                continue

            train = returns.iloc[:t]      # up to t-1
            high_tr  = high.iloc[:t]
            regime = "high" if bool(high.iloc[t - 1]) else "low"  # no lookahead

            # refit params every k_refit days (or first time)
            if (step % config.ROLLING_WINDOW == 0) or (params_low is None) or (params_high is None):
                low_train  = train[~high_tr]
                high_train = train[high_tr]

                fit_low  = low_train  if len(low_train)  >= MIN_OBS else train
                fit_high = high_train if len(high_train) >= MIN_OBS else train

                params_low  = self._am(fit_low,  "low").fit(disp="off").params
                params_high = self._am(fit_high, "high").fit(disp="off").params

            params = params_high if regime == "high" else params_low
            fixed  = self._am(train, regime).fix(params)
            fcst   = fixed.forecast(horizon=1, method=("simulation" if regime == "high" else "analytic"))
            v      = float(fcst.variance.iloc[-1, 0])

            preds.append(float(np.sqrt(max(v, 0.0))))
            dates.append(returns.index[t])

        pred_vol = pd.Series(preds, index=dates)

        idx2 = pred_vol.index.intersection(rv.index)
        pred_vol, rv2 = pred_vol.loc[idx2], rv.loc[idx2]

        mae  = float((pred_vol - rv2).abs().mean())
        rmse = float(((pred_vol - rv2) ** 2).mean() ** 0.5)

        return {
            "dates":      [str(d.date()) for d in pred_vol.index],
            "pred_vol":   [round(float(v), 4) for v in pred_vol.tolist()],
            "actual_vol": [round(float(v), 4) for v in rv2.tolist()],
            "mae":        round(mae, 4),
            "rmse":       round(rmse, 4),
            "duration":   int(duration),
        }