# Processing/src/phase4/sentiment_signal.py

import numpy as np
import pandas as pd
from typing import Callable, Tuple, Sequence

def zscore(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    return (x - x.mean()) / (x.std(ddof=0) + 1e-12)

def winsorize(s: pd.Series, p: float = 0.01) -> pd.Series:
    lo, hi = s.quantile(p), s.quantile(1 - p)
    return s.clip(lo, hi)

def ema(s: pd.Series, span: int = 5) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def link_exp(z: np.ndarray, lam0: float, beta: float, lam_min: float, lam_max: float) -> np.ndarray:
    out = lam0 * np.exp(beta * z)
    return np.clip(out, lam_min, lam_max)

def build_lambda_curve_from_features(
    df: pd.DataFrame,
    date_col: str = "date",
    driver_col: str = "sent_score_mean",   # or use model prob: "prob_up"
    smooth_span: int = 5,                  # EMA smoothing (trading days)
    lam0: float = 8.0,                     # baseline annual intensity
    beta: float = 0.5,                     # sensitivity
    lam_min: float = 0.5,                  # caps to keep MC stable
    lam_max: float = 40.0,
) -> Tuple[np.ndarray, np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    """
    Returns (tgrid_years, lambda_series, lambda_fn), where lambda_fn(t) is stepwise-constant.
    The grid is daily at 1/252 increments and aligned to df[date].
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    # Driver transformation: winsorize → zscore → EMA
    s = winsorize(df[driver_col].astype(float))
    z = zscore(s).fillna(0.0)
    z_sm = ema(z, span=smooth_span).fillna(method="bfill").fillna(0.0)

    lam_series = link_exp(z_sm.values, lam0, beta, lam_min, lam_max)

    # Daily business-day grid: 0, 1/252, 2/252, ...
    tgrid = np.arange(len(df), dtype=float) / 252.0

    from .sentiment_signal import make_step_fn  # already present in your file
    lambda_fn = make_step_fn(tgrid, lam_series)

    return tgrid, lam_series, lambda_fn

def make_step_fn(tgrid: np.ndarray, values: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    tgrid = np.asarray(tgrid, dtype=float)
    values = np.asarray(values, dtype=float)
    def fn(t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        idx = np.minimum(np.searchsorted(tgrid, t, side="right")-1, len(values)-1)
        idx = np.maximum(idx, 0)
        return values[idx]
    return fn
