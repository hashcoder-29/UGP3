
"""
Stage 2 calibration: with core Kou params fixed (from Stage 1), fit intensity dynamics (kappa, a, b, nu, lambda0)
by minimizing option pricing error versus market prices, using Monte Carlo pricer.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Sequence, Callable, Dict, Any, Tuple
from .cox_kou_mc import DiffusionParams, KouJumpParams, IntensityDynamics, price_european_mc

@dataclass
class MarketOption:
    K: float
    T: float
    is_call: bool
    mid: float
    weight: float = 1.0

def rmse(prices_model: np.ndarray, prices_mkt: np.ndarray, weights: np.ndarray | None=None) -> float:
    if weights is None:
        weights = np.ones_like(prices_model)
    err = (prices_model - prices_mkt)
    return float(np.sqrt(np.average(err**2, weights=weights)))

def calibrate_intensity(
    S0: float, r: float, diffusion: DiffusionParams, jump: KouJumpParams,
    options: Sequence[MarketOption],
    sentiment_fn: Callable[[np.ndarray], np.ndarray] | None,
    x0: Tuple[float,float,float,float,float] = (1.0, 0.1, 1.0, 0.1, 1.0),  # kappa, a, b, nu, lambda0
    bounds: Tuple[Tuple[float,float], ...] = ((1e-5, 10.0), (-10.0, 10.0), (-50.0, 50.0), (1e-6, 10.0), (1e-6, 50.0)),
    n_paths: int = 30_000, n_steps: int = 252, seed: int | None = 123
) -> Tuple[IntensityDynamics, float]:
    try:
        from scipy.optimize import minimize
    except Exception as e:
        raise RuntimeError("scipy is required for calibration") from e

    prices_mkt = np.array([opt.mid for opt in options], dtype=float)
    weights = np.array([opt.weight for opt in options], dtype=float)

    def obj(x):
        kappa, a, b, nu, lambda0 = x
        inten = IntensityDynamics(kappa=kappa, a=a, b=b, nu=nu, lambda0=lambda0)
        prices_model = np.array([
            price_european_mc(S0, opt.K, opt.T, r, diffusion, jump, inten, n_paths=n_paths, n_steps=n_steps, call=opt.is_call, sentiment_fn=sentiment_fn, seed=seed)
            for opt in options
        ], dtype=float)
        return rmse(prices_model, prices_mkt, weights)

    res = minimize(obj, np.array(x0, dtype=float), bounds=bounds, method="L-BFGS-B")
    xstar = res.x
    inten = IntensityDynamics(kappa=float(xstar[0]), a=float(xstar[1]), b=float(xstar[2]), nu=float(xstar[3]), lambda0=float(xstar[4]))
    return inten, float(res.fun)
