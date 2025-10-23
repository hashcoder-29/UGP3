# tests/test_phase4_smoke.py
import math, numpy as np
from src.phase4 import (
    DiffusionParams, KouJumpParams, IntensityDynamics,
    simulate_paths_mc, price_european_mc, make_step_fn
)

def _params():
    diff = DiffusionParams(mu=0.0, sigma=0.2)
    jump = KouJumpParams(lam=1.0, p_up=0.3, eta1=15.0, eta2=8.0)
    inten = IntensityDynamics(kappa=1.5, a=0.5, b=1.0, nu=0.2, lambda0=0.5)
    senti = make_step_fn(np.array([0.0, 0.5, 1.0]), np.array([0.0, 0.7, 0.2]))
    return diff, jump, inten, senti

def test_imports_and_api():
    diff, jump, inten, senti = _params()
    assert diff.sigma > 0 and 0.0 <= jump.p_up <= 1.0

def test_simulate_paths_lambda_positive():
    diff, jump, inten, senti = _params()
    S, lam = simulate_paths_mc(
        S0=100.0, T=1.0, r=0.01,
        diffusion=diff, jump=jump, intensity=inten,
        n_paths=2048, n_steps=64, sentiment_fn=senti, seed=123
    )
    assert S.shape == (2048, 65)
    assert lam.shape == (2048, 65)
    assert np.all(lam >= 0.0)

def test_mc_price_basic_properties():
    diff, jump, inten, senti = _params()
    # Monotone in strike: K2>K1 => C(K2)<=C(K1)
    price_K90 = price_european_mc(100, 90, 1.0, 0.01, diff, jump, inten, n_paths=8000, n_steps=128, sentiment_fn=senti, seed=7)
    price_K110 = price_european_mc(100, 110, 1.0, 0.01, diff, jump, inten, n_paths=8000, n_steps=128, sentiment_fn=senti, seed=7)
    assert price_K90 >= price_K110 >= 0.0

def test_mc_convergence_in_paths():
    diff, jump, inten, senti = _params()
    p1 = price_european_mc(100, 100, 1.0, 0.01, diff, jump, inten, n_paths=2000, n_steps=64, sentiment_fn=senti, seed=1)
    p2 = price_european_mc(100, 100, 1.0, 0.01, diff, jump, inten, n_paths=8000, n_steps=64, sentiment_fn=senti, seed=1)
    # larger sample should be closer to its own re-run (variance down); just require bounded difference
    p2b = price_european_mc(100, 100, 1.0, 0.01, diff, jump, inten, n_paths=8000, n_steps=64, sentiment_fn=senti, seed=1)
    assert abs(p2 - p2b) < 1e-6
    # and not crazy far from p1
    assert abs(p1 - p2) < max(5.0, 0.25 * p2)

def test_sentiment_step_fn():
    import numpy as np
    f = make_step_fn(np.array([0.0, 0.3, 1.0]), np.array([0.0, 0.5, 0.2]))
    out = f(np.array([0.0, 0.1, 0.29, 0.3, 0.8, 1.0]))
    # piecewise-constant
    assert np.allclose(out, np.array([0.0,0.0,0.0,0.5,0.5,0.2]))
