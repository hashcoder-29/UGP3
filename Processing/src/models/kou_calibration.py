from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Callable
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from numpy.fft import fft, ifft, fftfreq
from ..pricing.kou_cf import charfn_kou
from ..utils.numerics import set_seed, clamp, default_fft_density_grid

DT = 1.0 / 252.0  # daily step

@dataclass
class CalibResult:
    params: Dict[str, float]
    loglik: float
    aic: float
    bic: float
    success: bool
    nfev: int
    message: str

def _pdf_from_cf_on_grid(charfn: Callable[[np.ndarray], np.ndarray],
                         x_grid: np.ndarray,
                         u_grid: np.ndarray) -> np.ndarray:
    phi = charfn(u_grid)
    du = u_grid[1] - u_grid[0]
    expo = np.exp(-1j * np.outer(u_grid, x_grid))
    f = (du / (2.0 * np.pi)) * (phi[:, None] * expo).sum(axis=0)
    f = np.maximum(f.real, 1e-300)
    return f

def _neg_loglik(theta_raw: np.ndarray,
                x: np.ndarray,
                x_grid: np.ndarray,
                u_grid: np.ndarray) -> float:
    mu = theta_raw[0]
    sigma = np.exp(theta_raw[1])
    lam = np.exp(theta_raw[2])
    p = 1.0 / (1.0 + np.exp(-theta_raw[3]))
    eta1 = 1.0 + np.exp(theta_raw[4])
    eta2 = np.exp(theta_raw[5])

    params_p = {"sigma": sigma, "lambda": lam, "p": p, "eta1": eta1, "eta2": eta2, "r": 0.0, "d": 0.0}
    def cf_daily(u):
        return np.exp(1j * u * mu * DT) * charfn_kou(u, DT, params_p)

    f_grid = _pdf_from_cf_on_grid(cf_daily, x_grid, u_grid)
    interp = interp1d(x_grid, f_grid, kind="linear", bounds_error=False, fill_value=1e-300)
    fx = interp(x)
    nll = -np.sum(np.log(fx))
    if not np.isfinite(nll):
        return 1e300
    return nll

def calibrate_mle(
    returns: np.ndarray,
    n_starts: int = 8,
    seed: int = 7,
    x_span: float = 0.15,
    Nu: int = 2**12,
) -> CalibResult:
    set_seed(seed)
    x = np.asarray(returns).astype(float)
    x = x[np.isfinite(x)]
    n = x.size

    x_grid = np.linspace(-x_span, x_span, Nu)
    u_max = np.pi * (Nu - 1) / x_span
    u_grid = np.linspace(-u_max, u_max, Nu)

    m = np.nanmean(x) / DT
    v = np.nanvar(x) / DT
    sigma0 = np.sqrt(max(v, 1e-6)) * 0.8
    lam0 = 6.0
    p0 = 0.35
    eta1_0 = 20.0
    eta2_0 = 18.0

    best = None
    bounds = [
        (-1.0, 1.0),
        (np.log(1e-4), np.log(2.0)),
        (np.log(1e-6), np.log(50.0)),
        (-6.0, 6.0),
        (np.log(1e-6), np.log(200.0)),
        (np.log(1e-6), np.log(200.0)),
    ]

    for k in range(n_starts):
        z = np.random.default_rng(seed + k)
        theta0 = np.array([
            m + 0.05 * z.standard_normal(),
            np.log(sigma0) + 0.2 * z.standard_normal(),
            np.log(lam0)  + 0.5 * z.standard_normal(),
            np.log(p0/(1-p0)) + 0.5 * z.standard_normal(),
            np.log(eta1_0-1.0) + 0.5 * z.standard_normal(),
            np.log(eta2_0) + 0.5 * z.standard_normal(),
        ])
        obj = lambda th: _neg_loglik(th, x, x_grid, u_grid)
        res = minimize(obj, theta0, method="L-BFGS-B", bounds=bounds, options=dict(maxiter=500))
        if (best is None) or (res.fun < best.fun):
            best = res

    th = best.x
    mu = th[0]
    sigma = np.exp(th[1])
    lam = np.exp(th[2])
    p = 1.0 / (1.0 + np.exp(-th[3]))
    eta1 = 1.0 + np.exp(th[4])
    eta2 = np.exp(th[5])

    k_params = {"mu": mu, "sigma": sigma, "lambda": lam, "p": p, "eta1": eta1, "eta2": eta2}
    loglik = -float(best.fun)
    k = 6
    aic = 2 * k - 2 * loglik
    bic = k * np.log(n) - 2 * loglik

    return CalibResult(
        params=k_params, loglik=loglik, aic=aic, bic=bic,
        success=bool(best.success), nfev=int(best.nfev), message=str(best.message)
    )
