# Causal features for the regime HMM: daily log return and 21-day realized
# vol on the equity index. Every row t uses prices up to t only. The scaler
# is split into fit/apply so we can fit on the training window alone.

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import HMM_CONFIG


def build_features(equity: pd.Series, vol_window: int=HMM_CONFIG.vol_window) -> pd.DataFrame:
    log_ret=np.log(equity / equity.shift(1))
    realized_vol=log_ret.rolling(vol_window).std()
    return pd.DataFrame({"log_ret": log_ret, "realized_vol": realized_vol}).dropna()


def fit_scaler(train: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    # mean/std from the training window only
    std=train.std(ddof=0).replace(0.0, 1.0)  # avoid divide-by-zero on constants
    return train.mean(), std


def apply_scaler(
    features: pd.DataFrame, mean: pd.Series, std: pd.Series
) -> pd.DataFrame:
    return (features - mean) / std
