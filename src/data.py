# data.py
# download price data, compute log returns + rolling vol feature

import numpy as np
import pandas as pd
import yfinance as yf

import config


def load(ticker=None, start=config.START_DATE, cutoff=config.TRAIN_CUTOFF):
    """Returns: df, df_train, df_test"""
    
    ticker = ticker or config.TICKER
    df = yf.download(ticker, start, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.dropna(inplace=True)

    df["LogReturn"] = np.log(df["Close"]).diff()
    df["pctLogReturn"] = df["LogReturn"] * config.SCALE   # scale for numeric stability
    df.dropna(inplace=True)

    df_train = df[df.index < cutoff]
    df_test  = df[df.index >= cutoff]

    return df, df_train, df_test


def build_features(df):
    """Returns: df["LogReturn", "rolling_vol"]"""

    rolling_vol = df["LogReturn"].rolling(config.ROLLING_WINDOW).std()
    rolling_vol.name = "rolling_vol"

    features = pd.concat([df["LogReturn"], rolling_vol], axis=1).dropna()
    return features
