"""
Carr–Madan FFT call pricer (vectorized, with Simpson weights and log-strike shift).

Returns central half of the strike grid for better numerical conditioning.
"""
from __future__ import annotations
from typing import Callable, Tuple
from src.utils.backend import xp, fft, maybe_jit, asnumpy

def _simpson_weights(N: int):
    w = xp.ones(N)
    # Simpson's rule weights on equally spaced grid: 1,4,2,4,...,4,1
    w = w.at[0].set(1.0) if hasattr(w, "at") else (w.__setitem__(0, 1.0) or w)
    w = w.at[-1].set(1.0) if hasattr(w, "at") else (w.__setitem__(-1, 1.0) or w)
    if N > 2:
        # set odd indices to 4, even indices (except endpoints) to 2
        idx = xp.arange(1, N - 1)
        vals = xp.where(idx % 2 == 1, 4.0, 2.0)
        if hasattr(w, "at"):
            w = w.at[1:-1].set(vals)
        else:
            w[1:-1] = asnumpy(vals)
    w=w/3.0
    return w

@maybe_jit
def _psi_transform(v, T: float, r: float, alpha: float, charfn: Callable):
    """
    Damped transform psi(v) used by Carr–Madan.
    """
    i = 1j
    # Evaluate CF at shifted argument v - i(alpha+1)
    denom = (alpha * alpha + alpha - v * v + i * (2.0 * alpha + 1.0) * v)
    return xp.exp(-r * T) * charfn(v - i * (alpha + 1.0)) / denom

def price_calls_fft(
    S0: float,
    T: float,
    r: float,
    charfn: Callable,
    alpha: float = 1.5,
    N: int = 2 ** 12,
    eta: float = 0.25,
) -> Tuple["xp.ndarray", "xp.ndarray"]:
    """
    Price European calls on a log-strike grid using Carr–Madan FFT.

    Parameters
    ----------
    S0 : float
    T : float
    r : float
    charfn : callable
        u -> phi_X(u), CF of log-returns under Q.
    alpha : float
        Damping parameter (>0).
    N : int
        FFT grid size (power of 2 recommended).
    eta : float
        Frequency grid spacing.

    Returns
    -------
    strikes : ndarray (numpy)
    prices  : ndarray (numpy)
    """
    v = xp.arange(N, dtype=xp.float64) * eta         # frequency grid
    dk = 2.0 * xp.pi / (N * eta)                     # log-strike spacing
    b = xp.pi / eta                                  # grid half-width in k
    k = -b + dk * xp.arange(N, dtype=xp.float64)     # log-strike grid

    w = _simpson_weights(N)
    psi = _psi_transform(v, T, r, alpha, charfn)

    # Shift + scaling for FFT
    g = psi * w * xp.exp(1j * v * b) * eta

    # FFT to get prices in log-strike space
    F = fft.fft(g)
    Ck = (S0 * xp.exp(-alpha * k) / xp.pi) * xp.real(F)

    # Keep central half of the strikes for numerical quality
    m0, m1 = N // 4, 3 * N // 4
    strikes = S0 * xp.exp(k[m0:m1])
    prices = Ck[m0:m1]

    # Convert to numpy arrays for downstream consumers (tests use numpy)
    import numpy as np
    return np.asarray(asnumpy(strikes)), np.asarray(asnumpy(prices))
