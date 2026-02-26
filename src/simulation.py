# simulation.py
# regime-aware Monte Carlo price simulation

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

import config


def simulate(
    regime_labels,
    cond_vols,
    last_price,
    horizon=None,
    n_paths=None,
    nu: float = None,          # degrees-of-freedom for Student-t HIGH-regime shocks
):
    """
    Simulate N price paths of length H using regime-conditional volatilities.

    LOW regime  → Gaussian innovations  (matches GARCH normal assumption)
    HIGH regime → Student-t innovations (matches EGARCH StudentsT assumption)

    Parameters
    ----------
    regime_labels : array-like of str  ('low' | 'high')
    cond_vols     : array of floats    (scaled conditional volatilities)
    last_price    : float
    horizon       : int
    n_paths       : int
    nu            : float — Student-t degrees of freedom; falls back to
                    config or 8 if not supplied; drift
    """
    np.random.seed(config.SEED)

    H  = horizon or len(regime_labels)
    N  = n_paths or config.N_SIM
    nu = nu if (nu is not None and nu > 2) else 8.0   # nu≤2 → infinite variance

    paths = np.zeros((N, H + 1))
    paths[:, 0] = last_price

    for t, regime in enumerate(regime_labels[:H]):
        vol   = cond_vols[t] / config.SCALE   # back to raw log-return scale
        drift = config.LOW_DRIFT if regime == "low" else config.HIGH_DRIFT

        if regime == "high":
            shock = student_t.rvs(df=nu, scale=vol, size=N)
        else:
            shock = np.random.normal(0, vol, N)

        paths[:, t + 1] = paths[:, t] * np.exp(drift + shock)

    return paths


def fan_chart(paths: np.ndarray) -> pd.DataFrame:
    bands = np.percentile(paths, config.PERCENTILES, axis=0)
    cols  = [f"p{p}" for p in config.PERCENTILES]
    return pd.DataFrame(bands.T, columns=cols)