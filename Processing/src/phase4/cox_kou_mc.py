"""
Cox-intensity Kou jump-diffusion with sentiment-driven mean-reverting intensity.

Discrete-time Euler-Maruyama:
  dS/S- = (mu_eff) dt + sigma dW_S + sum_jumps (V_i - 1),  with  log V_i ~ Kou(double-exp)
  dλ = κ (θ(t) - λ) dt + ν dW_λ,    θ(t) = a + b * sentiment(t)

Risk-neutral drift correction:
  mu_eff(t) = (r - d) - λ_t * k_jump,
  k_jump = E[V-1] = E[e^Y] - 1,   Y ~ Kou(p_up, eta1, eta2)
  where E[e^Y] = p_up * eta1/(eta1-1) + (1-p_up) * eta2/(eta2+1),  (eta1>1)
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Callable, Tuple

Array = np.ndarray

@dataclass
class KouJumpParams:
    # Stage 1 baseline fields kept for compatibility; lam is ignored in Phase 4 path state (lambda_t is simulated)
    lam: float
    p_up: float
    eta1: float
    eta2: float

@dataclass
class DiffusionParams:
    # Set mu = r - d (risk-neutral). If you have dividends, use d>0 when building mu upstream.
    mu: float
    sigma: float

@dataclass
class IntensityDynamics:
    kappa: float
    a: float
    b: float
    nu: float
    lambda0: float
    # Optional correlation between price and intensity Brownian shocks
    rho_sl: float = 0.0
    # Optional safety caps for positivity and numerical stability
    lam_min: float = 1e-8
    lam_max: float = np.inf

def jump_compensator(p_up: float, eta1: float, eta2: float) -> float:
    """
    k_jump = E[e^Y] - 1 for the Kou double-exponential with P(up)=p_up.
    Requires eta1 > 1 for finiteness.
    """
    if eta1 <= 1.0:
        raise ValueError("eta1 must be > 1 for E[e^Y] to exist.")
    EeY = p_up * (eta1 / (eta1 - 1.0)) + (1.0 - p_up) * (eta2 / (eta2 + 1.0))
    return EeY - 1.0

def sample_kou_jump(rng: np.random.Generator, p_up: float, eta1: float, eta2: float, size=None) -> Array:
    """Draw Y=log(V) from asymmetric double-exponential (up: +Exp(eta1), down: -Exp(eta2))."""
    u = rng.random(size=size)
    is_up = (u < p_up)
    y = np.empty_like(u)
    y[is_up]  = rng.exponential(scale=1.0/eta1, size=is_up.sum())
    y[~is_up] = -rng.exponential(scale=1.0/eta2, size=(~is_up).sum())
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
    Returns (S_paths, lambda_paths) with shapes (n_paths, n_steps+1).
    Uses risk-neutral drift mu_eff(t) = diffusion.mu - lambda_t * k_jump.
    If you set diffusion.mu = r - d upstream, this enforces risk-neutral pricing.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps

    # Time grid and sentiment → theta(t)
    tgrid = np.linspace(0.0, T, n_steps+1)
    if sentiment_fn is None:
        theta_t = np.full(n_steps+1, intensity.a, dtype=float)
    else:
        sent = sentiment_fn(tgrid)  # shape (n_steps+1,)
        theta_t = intensity.a + intensity.b * sent

    S = np.full((n_paths, n_steps+1), S0, dtype=float)
    lam = np.full((n_paths, n_steps+1), intensity.lambda0, dtype=float)

    # Brownian increments (correlated if rho_sl != 0)
    if abs(intensity.rho_sl) > 0:
        z1 = rng.standard_normal(size=(n_paths, n_steps))
        z2 = rng.standard_normal(size=(n_paths, n_steps))
        dW_S = z1 * np.sqrt(dt)
        dW_l = (intensity.rho_sl * z1 + np.sqrt(1.0 - intensity.rho_sl**2) * z2) * np.sqrt(dt)
    else:
        dW_S = rng.normal(0.0, np.sqrt(dt), size=(n_paths, n_steps))
        dW_l = rng.normal(0.0, np.sqrt(dt), size=(n_paths, n_steps))

    mu_const, sigma = diffusion.mu, diffusion.sigma
    k_jump = jump_compensator(jump.p_up, jump.eta1, jump.eta2)

    for k in range(n_steps):
        # OU-style stochastic intensity with caps to ensure positivity/stability
        lam_k = lam[:, k]
        lam_next = lam_k + intensity.kappa * (theta_t[k] - lam_k) * dt + intensity.nu * dW_l[:, k]
        lam[:, k+1] = np.clip(lam_next, intensity.lam_min, intensity.lam_max)

        # Poisson arrivals with current intensity
        Nj = rng.poisson(lam_k * dt)

        # Sum of log-jumps this step (vectorized per-path accumulation)
        if Nj.max() == 0:
            logJ = np.zeros(n_paths, dtype=float)
        else:
            logJ = np.zeros(n_paths, dtype=float)
            idx = np.where(Nj > 0)[0]
            if idx.size:
                total = int(Nj[idx].sum())
                Y = sample_kou_jump(rng, jump.p_up, jump.eta1, jump.eta2, size=total)
                off = 0
                for i, c in zip(idx, Nj[idx]):
                    logJ[i] = Y[off:off+c].sum()
                    off += c

        # Risk-neutral drift correction: subtract λ_t * k_jump
        mu_eff = mu_const - lam_k * k_jump  # elementwise per path

        # Log-Euler update
        S[:, k+1] = S[:, k] * np.exp((mu_eff - 0.5 * sigma**2) * dt + sigma * dW_S[:, k] + logJ)

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
    return float(np.exp(-r * T) * payoff.mean())
