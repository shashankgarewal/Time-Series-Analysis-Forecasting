# config.py

TICKER = "MSFT"
START_DATE = "2014-01-01"
TRAIN_CUTOFF = "2022-01-01"
SCALE = 100  # multiply log returns for GARCH numeric stability

# feature
ROLLING_VOL_WINDOW = 21

# hysteresis regime detector
HYST_VOL_WINDOW = 21
HYST_LOW_TO_HIGH = 0.90   # quantile to enter HIGH regime
HYST_HIGH_TO_LOW = 0.75   # quantile to exit HIGH regime
HYST_PERSISTENCE = 20     # min consecutive days to confirm regime

# HMM regime detector
HMM_N_STATES = 2
HMM_N_ITER = 1000
HMM_SEED = 42

# vol models
P, Q = 1, 1
LOW_VOL  = "GARCH"
HIGH_VOL = "EGARCH"
HIGH_DIST = "StudentsT"

# simulation
N_SIM = 1000
HORIZON = 63        # ~3 months
SIM_SEED = 42
LOW_DRIFT  = 0.10 / 252   # 10% annual in calm regime
HIGH_DRIFT = 0.00 / 252   # flat in volatile regime
PERCENTILES = [2.5, 10, 50, 90, 97.5]

# api
HOST = "127.0.0.0"
PORT = 5000