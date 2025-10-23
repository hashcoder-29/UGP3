
"""
Cox-intensity Kou jump-diffusion with sentiment-driven mean-reverting intensity.
SDEs (discrete-time Euler-Maruyama):
    dS/S- = mu dt + sigma dW + sum_{jumps} (V_i - 1)
    dlambda = kappa * (theta(t) - lambda) dt + nu dW_lambda
    theta(t) = a + b * sentiment(t)
Pricing: Monte Carlo for European calls/puts.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Callable, Tuple

Array = np.ndarray

@dataclass
class KouJumpParams:
    lam: float          # baseline jump intensity (used for Stage 1; in Phase 4 this is the state lambda_t)
    p_up: float         # prob of upward jump (p), else q = 1-p
    eta1: float         # positive jump (log V >= 0) rate
    eta2: float         # negative jump (log V < 0) rate

@dataclass
class DiffusionParams:
    mu: float
    sigma: float

@dataclass
class IntensityDynamics:
    kappa: float
    a: float
    b: float
    nu: float
    lambda0: float

def sample_kou_jump(log_jump_size_rng: np.random.Generator, p_up: float, eta1: float, eta2: float, size=None) -> Array:
    """Draw Y=log(V) from asymmetric double-exponential."""
    u = log_jump_size_rng.random(size=size)
    is_up = (u < p_up)
    # draw exponential magnitudes
    y = np.empty_like(u)
    # For y>=0 with rate eta1: density eta1 * exp(-eta1 y)
    y[is_up] = log_jump_size_rng.exponential(scale=1.0/eta1, size=is_up.sum())
    # For y<0 with rate eta2: density eta2 * exp(eta2 y) for y<0 -> -Exp rate eta2
    y[~is_up] = -log_jump_size_rng.exponential(scale=1.0/eta2, size=(~is_up).sum())
    return y

def simulate_paths_mc(
    S0: float,
    T: float,
    r: float,
    diffusion: DiffusionParams,
    jump: KouJumpParams,
    intensity: IntensityDynamics,
    n_paths: int = 20_000,
    n_steps: int = 252,
    sentiment_fn: Callable[[Array], Array] | None = None,
    seed: int | None = 42,
) -> Tuple[Array, Array]:
    """
    Simulate asset price paths and intensity paths.
    Returns (S_paths, lambda_paths) with shapes (n_paths, n_steps+1).
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    S = np.full((n_paths, n_steps+1), S0, dtype=float)
    lam = np.full((n_paths, n_steps+1), intensity.lambda0, dtype=float)

    # Brownian increments
    dW_S = rng.normal(0.0, np.sqrt(dt), size=(n_paths, n_steps))
    dW_l = rng.normal(0.0, np.sqrt(dt), size=(n_paths, n_steps))

    # sentiment over grid
    tgrid = np.linspace(0.0, T, n_steps+1)
    if sentiment_fn is None:
        theta_series = intensity.a + intensity.b * 0.0
        theta_t = np.full(n_steps+1, theta_series, dtype=float)
    else:
        sent = sentiment_fn(tgrid)  # shape (n_steps+1,)
        theta_t = intensity.a + intensity.b * sent

    mu, sigma = diffusion.mu, diffusion.sigma

    for k in range(n_steps):
        # intensity OU-style update
        lam[:, k+1] = lam[:, k] + intensity.kappa * (theta_t[k] - lam[:, k]) * dt + intensity.nu * dW_l[:, k]
        lam[:, k+1] = np.clip(lam[:, k+1], 1e-8, None)  # keep positive

        # number of jumps ~ Poisson(lam*dt)
        Nj = rng.poisson(lam[:, k] * dt)
        # sum of log jump sizes in step k for each path
        if Nj.max() == 0:
            logJ = np.zeros(n_paths)
        else:
            logJ = np.zeros(n_paths)
            # For efficiency, sample in blocks where Nj>0
            idx = np.where(Nj > 0)[0]
            for i in idx:
                y = sample_kou_jump(rng, jump.p_up, jump.eta1, jump.eta2, size=Nj[i])
                logJ[i] = y.sum()

        # diffusion + jumps (risk-neutral drift r minus jump compensator is not enforced here; caller controls mu)
        S[:, k+1] = S[:, k] * np.exp((mu - 0.5*sigma**2) * dt + sigma * dW_S[:, k] + logJ)

    return S, lam

def price_european_mc(
    S0: float, K: float, T: float, r: float,
    diffusion: DiffusionParams, jump: KouJumpParams, intensity: IntensityDynamics,
    n_paths: int = 50_000, n_steps: int = 252, call: bool = True,
    sentiment_fn: Callable[[Array], Array] | None = None, seed: int | None = 42
) -> float:
    S, _ = simulate_paths_mc(S0, T, r, diffusion, jump, intensity, n_paths, n_steps, sentiment_fn, seed)
    ST = S[:, -1]
    payoff = np.maximum(ST - K, 0.0) if call else np.maximum(K - ST, 0.0)
    return float(np.exp(-r*T) * payoff.mean())
