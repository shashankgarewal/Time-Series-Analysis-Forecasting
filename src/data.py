# data.py
# download price data, compute log returns + rolling vol feature

import numpy as np
import pandas as pd
import yfinance as yf

import config


def load(ticker=None, start=config.START_DATE):
    ticker = ticker or config.TICKER
    """Returns: df, df_train, df_test"""
    df = yf.download(ticker, start, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.dropna(inplace=True)

    df["LogReturn"] = np.log(df["Close"]).diff()
    df["pctLogReturn"] = df["LogReturn"] * config.SCALE   # scale for numeric stability
    df.dropna(inplace=True)

    df_train = df[df.index < config.TRAIN_CUTOFF]
    df_test  = df[df.index >= config.TRAIN_CUTOFF]

    return df, df_train, df_test


def build_features(df):
    """Returns: df["LogReturn", "rolling_vol"]"""

    rolling_vol = df["LogReturn"].rolling(config.ROLLING_WINDOW).std()
    rolling_vol.name = "rolling_vol"

    features = pd.concat([df["LogReturn"], rolling_vol], axis=1).dropna()
    return features
