"""
Stage 2 calibration: with core Kou params fixed (from Stage 1), fit intensity dynamics
(kappa, a, b, nu, lambda0) by minimizing option pricing error vs market prices (MC).
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Sequence, Callable, Tuple, Optional
from .cox_kou_mc import DiffusionParams, KouJumpParams, IntensityDynamics, price_european_mc

@dataclass
class MarketOption:
    K: float
    T: float            # years to expiry from quote date
    is_call: bool
    mid: float
    weight: float = 1.0

def rmse(prices_model: np.ndarray, prices_mkt: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
    if weights is None:
        weights = np.ones_like(prices_model)
    err = prices_model - prices_mkt
    return float(np.sqrt(np.average(err * err, weights=weights)))

def calibrate_intensity(
    S0: float,
    r: float,
    diffusion: DiffusionParams,             # set diffusion.mu = r - d upstream
    jump: KouJumpParams,                    # Stage-1 Kou (eta1>1, etc.)
    options: Sequence[MarketOption],
    sentiment_fn: Callable[[np.ndarray], np.ndarray] | None,
    # x0: (kappa, a, b, nu, lambda0)
    x0: Tuple[float, float, float, float, float] = (2.0, 8.0, 4.0, 2.0, 8.0),
    # bounds:          kappa      a           b           nu         lambda0
    bounds: Tuple[Tuple[float, float], ...] = ((1e-3, 15.0), (0.10, 40.0), (-20.0, 20.0), (0.0, 15.0), (0.10, 40.0)),
    # MC controls
    n_paths: int = 30_000,
    n_steps: int = 252,
    # objective smoothing
    use_crn: bool = True,
    base_seed: int = 777,
    # intensity extras
    lam_min: float = 1e-8,
    lam_max: float = 60.0,
    rho_sl: float = 0.0,
    # mild regularization to discourage crazy params (set to 0.0 to disable)
    reg_coeff: float = 0.0,   # e.g., 1e-4
) -> Tuple[IntensityDynamics, float]:
    """
    Returns (best_intensity_params, best_rmse).
    """
    try:
        from scipy.optimize import minimize
    except Exception as e:
        raise RuntimeError("scipy is required for calibration") from e

    prices_mkt = np.array([opt.mid for opt in options], dtype=float)
    weights = np.array([opt.weight for opt in options], dtype=float)

    # CRN seeds: fixed across the whole optimization for a smooth objective.
    if use_crn:
        rng = np.random.default_rng(base_seed)
        per_opt_seed = rng.integers(1, 2**31 - 1, size=len(options), dtype=np.int64)
    else:
        per_opt_seed = np.array([base_seed] * len(options), dtype=np.int64)

    def obj(x: np.ndarray) -> float:
        kappa, a, b, nu, lambda0 = map(float, x)
        inten = IntensityDynamics(
            kappa=kappa, a=a, b=b, nu=nu, lambda0=lambda0,
            rho_sl=rho_sl, lam_min=lam_min, lam_max=lam_max
        )
        model = np.empty(len(options), dtype=float)
        for i, opt in enumerate(options):
            # keep seeds fixed across evaluations
            model[i] = price_european_mc(
                S0=S0, K=opt.K, T=opt.T, r=r,
                diffusion=diffusion, jump=jump, intensity=inten,
                n_paths=n_paths, n_steps=n_steps, call=opt.is_call,
                sentiment_fn=sentiment_fn, seed=int(per_opt_seed[i])
            )
        loss = rmse(model, prices_mkt, weights)
        if reg_coeff > 0.0:
            # small L2 penalty on b and nu and deviation of kappa from a moderate value
            loss += reg_coeff * ((b / 10.0) ** 2 + (nu / 5.0) ** 2 + ((kappa - 2.0) / 5.0) ** 2)
        return loss

    res = minimize(obj, np.array(x0, dtype=float), bounds=bounds, method="L-BFGS-B")
    kappa, a, b, nu, lambda0 = map(float, res.x)
    best = IntensityDynamics(
        kappa=kappa, a=a, b=b, nu=nu, lambda0=lambda0,
        rho_sl=rho_sl, lam_min=lam_min, lam_max=lam_max
    )
    return best, float(res.fun)
