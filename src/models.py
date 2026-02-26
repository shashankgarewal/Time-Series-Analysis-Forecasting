# models.py
# regime-conditional vol models: GARCH(1,1) for LOW, EGARCH(1,1)+t for HIGH

import numpy as np
import pandas as pd
from arch import arch_model

import config


class RegimeVolModel:
    def fit(self, pctLogReturn, regime_label):
        # align
        idx = pctLogReturn.index.intersection(regime_label.index)
        r = pctLogReturn.loc[idx]

        low_returns  = r[~regime_label.loc[idx]]
        high_returns = r[regime_label.loc[idx]]

        self.garch_fit = arch_model(
            low_returns, mean="Zero", vol="GARCH", p=config.P, q=config.Q, dist="normal"
        ).fit(disp="off")

        self.egarch_fit = arch_model(
            high_returns, mean="Zero", vol="EGARCH", p=config.P, q=config.Q, dist=config.HIGH_DIST
        ).fit(disp="off")

        return self

    def forecast_next(self, curr_regime):
        """
        returns: next day variance
        """
        if curr_regime == "high":
            am = self.egarch_fit
        else:
            am = self.garch_fit
            
        pred_var = float(am.forecast(horizon=1).variance.iloc[-1, 0])
        return float(np.sqrt(pred_var))

    def evaluate(self, pctLogReturn, high_mask, duration):
        # walk-forward vol forecast over the last `duration` days
        # at each step: train on everything before t, forecast t+1
        idx     = pctLogReturn.index.intersection(high_mask.index)
        returns = pctLogReturn.loc[idx]
        high    = high_mask.loc[idx]

        preds = []
        for step in range(duration):
            t      = len(returns) - duration + step
            train  = returns.iloc[:t]
            regime = "high" if bool(high.iloc[t]) else "low"

            if regime == "high":
                am = arch_model(train, mean="Zero", vol="EGARCH",
                                p=config.P, q=config.Q, dist=config.HIGH_DIST)
            else:
                am = arch_model(train, mean="Zero", vol="GARCH",
                                p=config.P, q=config.Q, dist="normal")

            fit      = am.fit(disp="off")
            pred_var = float(fit.forecast(horizon=1).variance.iloc[-1, 0])
            preds.append(float(np.sqrt(pred_var)))

        pred_vol = pd.Series(preds, index=returns.index[-duration:])

        # realised vol proxy: rolling rms of squared returns (5-day window)
        rv = (returns ** 2).rolling(config.ROLLING_WINDOW).mean() ** 0.5
        rv = rv.loc[pred_vol.index].dropna()

        # align both to same index
        idx2     = pred_vol.index.intersection(rv.index)
        pred_vol = pred_vol.loc[idx2]
        rv       = rv.loc[idx2]

        mae  = float((pred_vol - rv).abs().mean())
        rmse = float(((pred_vol - rv) ** 2).mean() ** 0.5)

        return {
            "dates":      [str(d.date()) for d in pred_vol.index],
            "pred_vol":   [round(v, 4) for v in pred_vol.tolist()],
            "actual_vol": [round(v, 4) for v in rv.tolist()],
            "mae":        round(mae, 4),
            "rmse":       round(rmse, 4),
        }