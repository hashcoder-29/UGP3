"""
MLE calibration of Kou parameters to daily log-returns using FFT-inverted densities.

Approach:
- For a parameter vector θ = {mu, sigma, lambda, p, eta1, eta2}, daily log-return X_{dt}
  has characteristic function φ_X(u; dt) from kou_cf.charfn_kou with T=dt and r,d=0 under P.
- We invert φ to a pdf on a fixed x-grid via iFFT once per likelihood evaluation.
- Use linear interpolation to evaluate f(x_i) for observed returns.
- Sum log f(x_i) -> maximize via SciPy with multi-start.

We keep dt = 1/252 for daily returns. We calibrate under P (physical), then for pricing
replace μ by risk-neutral drift via martingale condition (handled in pricing step).

Outputs:
- params dict with keys: mu, sigma, lambda, p, eta1, eta2
- log-likelihood, AIC, BIC
"""
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
    """
    Invert CF to PDF on x_grid using Fourier transform pairing.
    We use:
      f(x) ≈ (1/(2π)) ∫ φ(u) e^{-i u x} du
    with evenly spaced u_grid and FFT normalization baked in.

    We construct symmetric u-grid and use numpy.fft conventions.
    """
    # φ(u) on symmetric grid
    phi = charfn(u_grid)
    # Inverse transform via FFT (note numpy.fft assumes sum exp(+i 2π k n /N))
    # We build grids to match a Riemann sum approximation:
    du = u_grid[1] - u_grid[0]
    # Compute f on the x grid by discrete inverse transform:
    # f(x_j) ≈ (1/2π) * sum_k φ(u_k) * exp(-i u_k x_j) du
    # Vectorized with outer product
    expo = np.exp(-1j * np.outer(u_grid, x_grid))  # shape (Nu, Nx)
    f = (du / (2.0 * np.pi)) * (phi[:, None] * expo).sum(axis=0)
    f = np.maximum(f.real, 1e-300)  # avoid log(0); ensure non-negative
    return f

def _neg_loglik(theta_raw: np.ndarray,
                x: np.ndarray,
                x_grid: np.ndarray,
                u_grid: np.ndarray) -> float:
    """
    Negative log-likelihood for returns x given raw param vector (with transforms):
    theta_raw = [mu, log_sigma, log_lambda, logit_p, log_eta1_shift, log_eta2]
    with eta1 = 1 + exp(log_eta1_shift) to enforce eta1>1.
    """
    mu = theta_raw[0]
    sigma = np.exp(theta_raw[1])
    lam = np.exp(theta_raw[2])
    p = 1.0 / (1.0 + np.exp(-theta_raw[3]))  # logistic
    eta1 = 1.0 + np.exp(theta_raw[4])
    eta2 = np.exp(theta_raw[5])

    params_p = dict(sigma=sigma, lambda=lam, p=p, eta1=eta1, eta2=eta2, r=0.0, d=0.0)
    # Characteristic function for X_dt under P: add drift mu in φ via shift:
    # For X = drift*dt + KouJumpDiff(dt), φ_X(u) = exp(i u mu dt) * φ_Kou(u; dt with r=d=0)
    def cf_daily(u):
        return np.exp(1j * u * mu * DT) * charfn_kou(u, DT, params_p)

    f_grid = _pdf_from_cf_on_grid(cf_daily, x_grid, u_grid)
    # Interpolate PDF on observed x
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
    """
    Calibrate Kou params to daily log-returns via MLE.

    Parameters
    ----------
    returns : np.ndarray
        Daily log-returns (np.log(S_t/S_{t-1})).
    n_starts : int
        Number of multi-start initializations.
    seed : int
        Deterministic seed.
    x_span : float
        Half-width of x-grid for density inversion (±x_span).
    Nu : int
        Number of points in u-grid (also x-grid pairs).

    Returns
    -------
    CalibResult
    """
    set_seed(seed)
    x = np.asarray(returns).astype(float)
    x = x[np.isfinite(x)]
    n = x.size
    # Build inversion grids
    x_grid = np.linspace(-x_span, x_span, Nu)
    # u-grid spacing consistent with x_grid resolution: Δu ≈ 2π / (x_range)
    u_max = np.pi * (Nu - 1) / x_span
    u_grid = np.linspace(-u_max, u_max, Nu)

    # Data moments for initial guesses
    m = np.nanmean(x) / DT
    v = np.nanvar(x) / DT
    sigma0 = np.sqrt(max(v, 1e-6)) * 0.8
    lam0 = 6.0  # jumps/year
    p0 = 0.35
    eta1_0 = 20.0
    eta2_0 = 18.0

    best = None
    bounds = [
        (-1.0, 1.0),          # mu (annualized drift) ~ [-100%, +100%]
        (np.log(1e-4), np.log(2.0)),   # log_sigma
        (np.log(1e-6), np.log(50.0)),  # log_lambda
        (-6.0, 6.0),          # logit_p
        (np.log(1e-6), np.log(200.0)), # log_eta1_shift (eta1>1 ensured)
        (np.log(1e-6), np.log(200.0)), # log_eta2
    ]

    for k in range(n_starts):
        # randomized starts around heuristics
        z = np.random.default_rng(seed + k)
        theta0 = np.array([
            m + 0.05 * z.standard_normal(),                 # mu
            np.log(sigma0) + 0.2 * z.standard_normal(),     # log_sigma
            np.log(lam0)  + 0.5 * z.standard_normal(),      # log_lambda
            np.log(p0/(1-p0)) + 0.5 * z.standard_normal(),  # logit_p
            np.log(eta1_0-1.0) + 0.5 * z.standard_normal(), # log_eta1_shift
            np.log(eta2_0) + 0.5 * z.standard_normal(),     # log_eta2
        ])

        obj = lambda th: _neg_loglik(th, x, x_grid, u_grid)
        res = minimize(obj, theta0, method="L-BFGS-B", bounds=bounds, options=dict(maxiter=500))
        if (best is None) or (res.fun < best.fun):
            best = res

    # Decode best params
    th = best.x
    mu = th[0]
    sigma = np.exp(th[1])
    lam = np.exp(th[2])
    p = 1.0 / (1.0 + np.exp(-th[3]))
    eta1 = 1.0 + np.exp(th[4])
    eta2 = np.exp(th[5])

    k_params = dict(mu=mu, sigma=sigma, lambda=lam, p=p, eta1=eta1, eta2=eta2)
    loglik = -float(best.fun)
    k = 6  # number of free parameters
    aic = 2 * k - 2 * loglik
    bic = k * np.log(n) - 2 * loglik

    return CalibResult(
        params=k_params, loglik=loglik, aic=aic, bic=bic,
        success=bool(best.success), nfev=int(best.nfev), message=str(best.message)
    )
