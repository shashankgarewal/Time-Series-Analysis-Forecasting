# config.py

# random state
SEED = 42

# data
TICKER = "MSFT"
START_DATE = "2014-01-01"
TRAIN_CUTOFF = "2022-01-01"
SCALE = 100  # multiply log returns for GARCH numeric stability

# feature
ROLLING_WINDOW = 21

# hysteresis regime detector
HYST_LOW_TO_HIGH = 0.80   # quantile to enter HIGH regime
HYST_HIGH_TO_LOW = 0.65   # quantile to exit HIGH regime
HYST_PERSISTENCE = 15     # min consecutive days to confirm regime

# HMM regime detector
HMM_N_STATES = 2
HMM_N_ITER = 1000

# vol models
P, Q = 1, 1
LOW_VOL  = "GARCH"
HIGH_VOL = "EGARCH"
HIGH_DIST = "StudentsT"

# simulation
N_SIM = 1000
HORIZON = 63        # ~3 months
LOW_DRIFT  = 0.10 / 252   # 10% annual in calm regime
HIGH_DRIFT = 0.00 / 252   # flat in volatile regime
PERCENTILES = [5, 25, 50, 75, 95]

# api
HOST = "0.0.0.0"   # FIX: was "127.0.0.0" which is not a valid bind address
PORT = 5000