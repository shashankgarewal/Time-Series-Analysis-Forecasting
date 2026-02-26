# regime.py
# two independent regime detectors: hysteresis-threshold and 2-state Gaussian HMM

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM

import config


# --- Hysteresis detector ---

class HysteresisDetector:
    """Hystersis Detector Class"""
    def fit(self, feature):
        """determine threshold for regime"""
        self.enter_thresh = feature.quantile(config.HYST_LOW_TO_HIGH)
        self.exit_thresh  = feature.quantile(config.HYST_HIGH_TO_LOW)
        return self

    def predict(self, feature):

        # hysteresis state machine
        states, in_high = [], False
        for v in feature.values:
            if not in_high and v >= self.enter_thresh:
                in_high = True
            elif in_high and v <= self.exit_thresh:
                in_high = False
            states.append(in_high)

        high_raw = pd.Series(states, index=feature.index, dtype=bool)

        # drop short HIGH bursts (persistence filter)
        m = high_raw.astype(int)
        grp = (m != m.shift()).cumsum()
        run_len = m.groupby(grp).transform("size")

        high = high_raw.copy()
        high[(high_raw == True) & (run_len < config.HYST_PERSISTENCE)] = False
        low = ~high

        return high, low


# --- HMM detector ---

class HMMDetector:
    """Class has fit(), predict() methods"""
    def fit(self, features):
        """features: [LogReturn, rolling_vol] DATAFRAME"""
        X = features.values

        self.model = GaussianHMM(
            n_components=config.HMM_N_STATES,
            covariance_type="full",
            n_iter=config.HMM_N_ITER,
            random_state=config.SEED,
            tol=1e-12
        )
        self.model.fit(X)

        # state with higher mean rolling_vol = HIGH
        self.high_state = int(np.argmax(self.model.means_[:, 1]))
        return self

    def predict(self, features):
        """
        features: [LogReturn, rolling_vol] DATAFRAMEs
        returns: DATAFRAMEs high, low regime 
        """
        X = features.values
        raw = self.model.predict(X)

        labels = pd.Series(
            ["high" if s == self.high_state else "low" for s in raw],
            index=features.index,
        )

        high = labels == "high"
        low  = labels == "low"
        return high, low
