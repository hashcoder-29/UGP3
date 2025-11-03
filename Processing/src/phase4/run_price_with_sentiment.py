# Processing/src/phase4/run_price_with_sentiment.py
"""
Phase 4 helper: price a single European option on a given quote/maturity date
under a news-driven Cox–Kou jump-diffusion, using the Monte Carlo pricer.

Notes
-----
- We build a time-varying jump intensity λ(t) from Phase-5 features via
  `build_lambda_curve_from_features`, parameterized by (lam0, beta).
- The MC pricer in `cox_kou_mc.py` expects separate objects for diffusion,
  jump-shape (Kou), and intensity dynamics; **it does not take a dividend `d`.**
  We keep `d` in this wrapper’s signature for backward compatibility but ignore it.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .cox_kou_mc import (
    DiffusionParams,
    KouJumpParams,
    IntensityDynamics,
    price_european_mc,
)
from .sentiment_signal import build_lambda_curve_from_features


def price_on_date(
    features_csv: str,
    quote_date: str,        # "YYYY-MM-DD"
    maturity_date: str,     # "YYYY-MM-DD"
    S0: float,
    r: float,
    d: float,               # kept for backward-compatibility; not used by the MC pricer
    K: float,
    is_call: bool,
    lam0: float,
    beta: float,
    kou_sigma: float,
    p_up: float,
    eta1: float,
    eta2: float,
    n_steps: Optional[int] = None,
    n_paths: int = 20_000,
    seed: Optional[int] = 42,
) -> float:
    """
    Price a European option using a Cox–Kou MC with news-driven λ(t).

    Parameters
    ----------
    features_csv : str
        Path to Phase-5 features CSV. Must contain a 'date' column and the
        driver column used by `build_lambda_curve_from_features` (default: 'sent_score_mean').
    quote_date : str
        Quote date "YYYY-MM-DD". Only features up to this date are used.
    maturity_date : str
        Expiry date "YYYY-MM-DD".
    S0 : float
        Underlying level at quote_date.
    r : float
        Risk-free continuously compounded rate.
    d : float
        Dividend yield (ignored by the current MC pricer).
    K : float
        Strike.
    is_call : bool
        True for call, False for put.
    lam0 : float
        Baseline (long-run) jump intensity.
    beta : float
        Sensitivity of intensity to the sentiment driver.
    kou_sigma : float
        Diffusion volatility (σ).
    p_up : float
        Kou up-jump probability parameter (p).
    eta1 : float
        Kou positive jump rate (η₁).
    eta2 : float
        Kou negative jump rate (η₂).
    n_steps : Optional[int]
        Time steps for MC. Defaults to business-day steps ~ 252*T.
    n_paths : int
        Number of Monte Carlo paths.
    seed : Optional[int]
        RNG seed.

    Returns
    -------
    float
        Model price.
    """
    # ---- Load features up to quote_date
    df = pd.read_csv(features_csv)
    if "date" not in df.columns:
        raise ValueError("features_csv must contain a 'date' column.")
    df["date"] = pd.to_datetime(df["date"])
    qd = pd.to_datetime(quote_date)
    df = df[df["date"] <= qd].sort_values("date")
    if df.empty:
        raise ValueError("No features available up to quote_date.")

    # ---- Build λ(t) from features (returns (tgrid, lambdas, lambda_fn))
    # Adjust driver_col to whatever you used in feature_builder (default below).
    _, _, lambda_fn = build_lambda_curve_from_features(
        df,
        date_col="date",
        driver_col="sent_score_mean",
        smooth_span=5,
        lam0=lam0,
        beta=beta,
        lam_min=0.5,
        lam_max=40.0,
    )

    # ---- Year fraction (business days / 252)
    T_days = int(np.busday_count(np.datetime64(quote_date, "D"), np.datetime64(maturity_date, "D")))
    T = max(T_days, 1) / 252.0

    # ---- Model parameters
    diffusion = DiffusionParams(mu=0.0, sigma=kou_sigma)
    # Set lam to lam0 here; time-variation comes from IntensityDynamics + lambda_fn
    jump = KouJumpParams(lam=lam0, p_up=p_up, eta1=eta1, eta2=eta2)
    # Simple mean-reverting intensity with affine news driver θ(t)=a+b*sent(t)
    intensity = IntensityDynamics(
        kappa=2.0,
        a=lam0,
        b=beta,
        nu=0.2,
        lambda0=lam0,
    )

    # ---- MC discretization
    steps = n_steps if n_steps is not None else max(int(round(T * 252)), 1)

    # ---- Price (the MC pricer ignores `d`; risk-neutral drift handled internally)
    price = price_european_mc(
        S0=S0,
        K=K,
        T=T,
        r=r,
        is_call=is_call,
        diffusion=diffusion,
        jump=jump,
        intensity=intensity,
        n_steps=steps,
        n_paths=n_paths,
        sentiment_fn=lambda_fn,
        seed=seed,
    )
    return float(price)
