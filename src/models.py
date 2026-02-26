# models.py
# regime-conditional vol models: GARCH(1,1) for LOW, EGARCH(1,1)+t for HIGH

import numpy as np
import pandas as pd
from arch import arch_model

import config


class RegimeVolModel:
    def fit(self, pctLogReturn, high_mask):
        # align
        idx = pctLogReturn.index.intersection(high_mask.index)
        r = pctLogReturn.loc[idx]

        low_returns  = r[~high_mask.loc[idx]]
        high_returns = r[high_mask.loc[idx]]

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

