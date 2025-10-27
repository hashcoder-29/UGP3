# Processing/src/phase4/run_price_with_sentiment.py

import pandas as pd
import numpy as np
from .cox_kou_mc import KouParams, price_european_mc
from .sentiment_signal import build_lambda_curve_from_features

def price_on_date(
    features_csv: str,
    quote_date: str,        # "YYYY-MM-DD"
    maturity_date: str,     # "YYYY-MM-DD"
    S0: float, r: float, d: float,
    K: float, is_call: bool,
    lam0: float, beta: float,
    kou_sigma: float, p_up: float, eta1: float, eta2: float,
    n_steps: int = 252, n_paths: int = 20000, seed: int | None = 42
) -> float:
    df = pd.read_csv(features_csv, parse_dates=["date"]).sort_values("date")
    df = df[df["date"] <= pd.to_datetime(quote_date)]
    if df.empty:
        raise ValueError("No features available up to quote_date.")

    _, _, lambda_fn = build_lambda_curve_from_features(
        df, date_col="date", driver_col="sent_score_mean",
        smooth_span=5, lam0=lam0, beta=beta, lam_min=0.5, lam_max=40.0
    )

    T_days = np.busday_count(np.datetime64(quote_date, 'D'), np.datetime64(maturity_date, 'D'))
    T = max(T_days, 1) / 252.0

    kou = KouParams(sigma=kou_sigma, p_up=p_up, eta1=eta1, eta2=eta2)
    price = price_european_mc(
        S0=S0, r=r, d=d, T=T, K=K, is_call=is_call,
        n_steps=max(int(T*252), 1), n_paths=n_paths,
        kou=kou, lambda_fn=lambda_fn, seed=seed
    )
    return price
