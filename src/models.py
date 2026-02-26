# models.py
# regime-conditional vol models: GARCH(1,1) for LOW, EGARCH(1,1)+t for HIGH

import numpy as np
import pandas as pd
from arch import arch_model

import config

MIN_OBS = 30   # minimum observations needed to fit a model


class RegimeVolModel:

    def fit(self, pct_returns: pd.Series, high_mask: pd.Series) -> "RegimeVolModel":
        idx = pct_returns.index.intersection(high_mask.index)
        r   = pct_returns.loc[idx]

        low_returns  = r[~high_mask.loc[idx]]
        high_returns = r[high_mask.loc[idx]]

        if len(low_returns) < MIN_OBS:
            raise ValueError(
                f"Not enough LOW-regime samples to fit GARCH "
                f"({len(low_returns)} < {MIN_OBS})"
            )
        if len(high_returns) < MIN_OBS:
            raise ValueError(
                f"Not enough HIGH-regime samples to fit EGARCH "
                f"({len(high_returns)} < {MIN_OBS}). "
                "Try lowering HYST_LOW_TO_HIGH threshold in config.py."
            )

        self.garch_fit = arch_model(
            low_returns, mean="Zero", vol="GARCH",
            p=config.P, q=config.Q, dist="normal"
        ).fit(disp="off")

        self.egarch_fit = arch_model(
            high_returns, mean="Zero", vol="EGARCH",
            p=config.P, q=config.Q, dist=config.HIGH_DIST
        ).fit(disp="off")

        return self

    def forecast_next(self, pct_returns: pd.Series, regime: str) -> float:
        if regime == "high":
            am     = arch_model(pct_returns, mean="Zero", vol="EGARCH", p=config.P, q=config.Q, dist=config.HIGH_DIST)
            params = self.egarch_fit.params
        else:
            am     = arch_model(pct_returns, mean="Zero", vol="GARCH",  p=config.P, q=config.Q, dist="normal")
            params = self.garch_fit.params

        fixed    = am.fix(params)
        pred_var = float(fixed.forecast(horizon=1).variance.iloc[-1, 0])
        return float(np.sqrt(max(pred_var, 0.0)))   # guard against tiny negatives

    @property
    def egarch_nu(self) -> float:
        """Degrees-of-freedom from the fitted Student-t EGARCH model."""
        params = self.egarch_fit.params
        # arch names the parameter 'nu' for StudentsT
        return float(params.get("nu", params.get("Nu", 8.0)))

    def evaluate_on_test(self, pct_test: pd.Series, high_test: pd.Series) -> dict:
        idx  = pct_test.index.intersection(high_test.index)
        r    = pct_test.loc[idx]
        high = high_test.loc[idx]

        # Both models run on the full contiguous return series
        garch_vol  = arch_model(r, mean="Zero", vol="GARCH",  p=config.P, q=config.Q, dist="normal") \
                         .fix(self.garch_fit.params).conditional_volatility
        egarch_vol = arch_model(r, mean="Zero", vol="EGARCH", p=config.P, q=config.Q, dist=config.HIGH_DIST) \
                         .fix(self.egarch_fit.params).conditional_volatility

        # Select vol adaptively: HIGH regime → EGARCH, LOW regime → GARCH
        pred_vol = pd.Series(
            [egarch_vol.loc[d] if high.loc[d] else garch_vol.loc[d] for d in r.index],
            index=r.index,
        )

        rv   = (r ** 2).rolling(5).mean() ** 0.5
        rv   = rv.dropna()
        idx2 = pred_vol.index.intersection(rv.index)
        pred_vol, rv = pred_vol.loc[idx2], rv.loc[idx2]

        mae  = float((pred_vol - rv).abs().mean())
        rmse = float(((pred_vol - rv) ** 2).mean() ** 0.5)

        return {
            "dates":      [str(d.date()) for d in pred_vol.index],
            "pred_vol":   [round(float(v), 4) for v in pred_vol],
            "actual_vol": [round(float(v), 4) for v in rv],
            "mae":        round(mae, 4),
            "rmse":       round(rmse, 4),
        }

    def walk_forward_eval(self, pct_returns: pd.Series, high_mask: pd.Series, duration: int) -> dict:
        """
        Walk-forward evaluation over the last `duration` steps (expanding window).
        Uses forecast_next() which now correctly uses fixed trained parameters.
        """
        idx     = pct_returns.index.intersection(high_mask.index)
        returns = pct_returns.loc[idx]
        high    = high_mask.loc[idx]

        preds = []
        for step in range(duration):
            t      = len(returns) - duration + step
            train  = returns.iloc[:t]
            regime = "high" if bool(high.iloc[t]) else "low"
            preds.append(self.forecast_next(train, regime))

        pred_vol = pd.Series(preds, index=returns.index[-duration:])

        rv   = (returns ** 2).rolling(5).mean() ** 0.5
        rv   = rv.loc[pred_vol.index].dropna()
        idx2 = pred_vol.index.intersection(rv.index)
        pred_vol, rv = pred_vol.loc[idx2], rv.loc[idx2]

        mae  = float((pred_vol - rv).abs().mean())
        rmse = float(((pred_vol - rv) ** 2).mean() ** 0.5)

        return {
            "dates":      [str(d.date()) for d in pred_vol.index],
            "pred_vol":   [round(float(v), 4) for v in pred_vol],
            "actual_vol": [round(float(v), 4) for v in rv],
            "mae":        round(mae, 4),
            "rmse":       round(rmse, 4),
            "duration":   duration,
        }