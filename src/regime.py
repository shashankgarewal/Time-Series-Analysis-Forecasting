# regime.py
# two independent regime detectors: hysteresis-threshold and 2-state Gaussian HMM

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

import config


# --- Hysteresis detector ---

class HysteresisDetector:
    """.fit() & .predict()"""
    def fit(self, feature: pd.Series) -> "HysteresisDetector":
        """Compute entry/exit quantile thresholds from training rolling_vol."""
        self.enter_thresh = feature.quantile(config.HYST_LOW_TO_HIGH)
        self.exit_thresh  = feature.quantile(config.HYST_HIGH_TO_LOW)
        return self

    def predict(self, feature: pd.Series):
        """
        Run hysteresis state machine over feature series.
        Returns (high: bool Series, low: bool Series).
        """
        states, in_high = [], False
        for v in feature.values:
            if not in_high and v >= self.enter_thresh:
                in_high = True
            elif in_high and v <= self.exit_thresh:
                in_high = False
            states.append(in_high)

        high_raw = pd.Series(states, index=feature.index, dtype=bool)

        # drop short HIGH bursts (persistence filter)
        m       = high_raw.astype(int)
        grp     = (m != m.shift()).cumsum()
        run_len = m.groupby(grp).transform("size")

        high = high_raw.copy()
        high[(high_raw == True) & (run_len < config.HYST_PERSISTENCE)] = False
        low  = ~high

        return high, low


# --- HMM detector ---

class HMMDetector:
    def fit(self, features: pd.DataFrame) -> "HMMDetector":
        """Fit HMM directly on [pctLogReturn, rolling_vol] features."""
        self.model = GaussianHMM(
            n_components=config.HMM_N_STATES, covariance_type="full", n_iter=config.HMM_N_ITER, 
            random_state=config.SEED, tol=1e-12)
        self.model.fit(features.values)

        # state with higher mean rolling_vol (col 1) = HIGH regime
        self.high_state = int(np.argmax(self.model.means_[:, 1]))
        return self

    def predict(self, features: pd.DataFrame):
        """
        Predict regime labels for feature DataFrame.
        Returns (high: bool Series, low: bool Series).
        """
        raw  = self.model.predict(features.values)
        high = pd.Series(raw == self.high_state, index=features.index, dtype=bool)
        low  = ~high
        return high, low