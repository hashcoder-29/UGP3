"""
Monte Carlo simulation of Kou price paths.

Note: baseline uses NumPy RNG for determinism & simplicity.
A JAX path can be added later using jax.random + jit/scan/vmap.
"""
from __future__ import annotations
from typing import Callable, Optional
import numpy as np
from src.utils.backend import USE_JAX  # hook only, keep NumPy RNG here

def _sample_double_exp_np(rng: np.random.Generator, size, p: float, eta1: float, eta2: float):
    flips = rng.random(size) < p
    y = np.empty(size, dtype=np.float64)
    # +Exp(eta1) for flips==True; -Exp(eta2) otherwise
    y[flips] = rng.exponential(scale=1.0 / eta1, size=flips.sum())
    y[~flips] = -rng.exponential(scale=1.0 / eta2, size=(~flips).sum())
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
):
    """
    Simulate price paths under GBM + compound Poisson with double-exponential jumps.

    Returns
    -------
    paths : ndarray, shape (n_paths, n_steps+1)
    """
    dt = T / n_steps
    rng = np.random.default_rng(seed)
    paths = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    paths[:, 0] = S0

    drift = (mu - 0.5 * sigma * sigma) * dt
    vol = sigma * np.sqrt(dt)

    t = 0.0
    for step in range(1, n_steps + 1):
        lam_t = float(lambda_func(t)) if lambda_func is not None else lam
        # diffusion
        z = rng.standard_normal(n_paths)
        incr = drift + vol * z

        # jumps: Poisson count per path
        Nj = rng.poisson(lam_t * dt, size=n_paths)
        has_jump = Nj > 0
        if np.any(has_jump):
            # total jump log-multiplier sum per path (sum of Y_j)
            Jsum = np.zeros(n_paths, dtype=np.float64)
            # process only paths that have jumps to keep it light
            idx = np.where(has_jump)[0]
            counts = Nj[idx]
            # sample variable number of jumps per path
            for k, c in zip(idx, counts):
                Y = _sample_double_exp_np(rng, c, p, eta1, eta2)
                Jsum[k] = Y.sum()
            incr[has_jump] += Jsum

        paths[:, step] = paths[:, step - 1] * np.exp(incr)
        t += dt

    return paths
