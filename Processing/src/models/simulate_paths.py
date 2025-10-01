"""
Monte Carlo paths for Kou jump-diffusion with optional time-varying intensity λ(t).

- Supports constant λ or a callable lambda_func(t) for Phase 4 plug-in (e.g., news driven).
- Vectorized over paths; uses numpy Generator with seed for determinism.

S_{t+dt} = S_t * exp((mu - 0.5 σ^2) dt + σ sqrt(dt) ε) * Π_{j=1}^{N_dt} V_j
Y = ln V ~ double-exponential with P(Y≥0)=p, upward tail Exp(η1); P(Y<0)=1-p, downward tail Exp(η2).
"""
from __future__ import annotations
import numpy as np
from typing import Optional, Callable, Dict

def _sample_double_exp(rng: np.random.Generator, size, p, eta1, eta2):
    u = rng.random(size)
    is_up = u < p
    y = np.empty(size, dtype=float)
    # Up jumps: y >= 0 with density η1 e^{-η1 y}
    y[is_up] = rng.exponential(1.0/eta1, is_up.sum())
    # Down jumps: y < 0 with density η2 e^{η2 y} -> -Exp(η2)
    y[~is_up] = -rng.exponential(1.0/eta2, (~is_up).sum())
    return y

def simulate_paths(
    S0: float,
    T: float,
    mu: float,
    sigma: float,
    lam: float,
    p: float,
    eta1: float,
    eta2: float,
    n_paths: int = 20000,
    n_steps: int = 252,
    seed: int = 42,
    lambda_func: Optional[Callable[[float], float]] = None,
) -> np.ndarray:
    """
    Simulate price paths under Kou model.

    Returns
    -------
    paths : np.ndarray, shape (n_paths, n_steps+1)
        Simulated price paths including S0 at column 0.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    times = np.linspace(0.0, T, n_steps+1)
    S = np.empty((n_paths, n_steps+1), dtype=float)
    S[:, 0] = S0
    for t_idx in range(n_steps):
        t = times[t_idx]
        lam_t = lam if lambda_func is None else float(lambda_func(t))
        # Diffusion term
        eps = rng.standard_normal(n_paths)
        diff = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * eps
        # Jumps: Poisson number then product of jump factors
        n_jumps = rng.poisson(lam_t * dt, n_paths)
        # If few jumps expected, generate only necessary Y's
        J = np.ones(n_paths, dtype=float)
        mask = n_jumps > 0
        idx = np.where(mask)[0]
        for i in idx:
            y = _sample_double_exp(rng, n_jumps[i], p, eta1, eta2)
            J[i] = np.exp(y.sum())
        S[:, t_idx+1] = S[:, t_idx] * np.exp(diff) * J
    return S
