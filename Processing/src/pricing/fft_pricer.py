"""
Carr–Madan (1999) FFT pricer for European calls using a provided characteristic function.

We price a damped call C_alpha(k) = e^{α k} C(k) with α>0. Its Fourier transform ψ(v) is:
 ψ(v) = e^{-rT} * φ_X( v - i(α+1) ) / (α^2 + α - v^2 + i(2α+1)v)

Numerical details:
- Simpson weights improve integration accuracy on the FFT grid.
- Expose N (grid size), η (frequency spacing), α (damping).
- Return strikes and prices aligned; users can interpolate to their desired K.

References: Carr & Madan (1999)
"""
from __future__ import annotations
import numpy as np
from numpy.fft import fft
from typing import Callable, Tuple

def price_calls_fft(
    S0: float,
    T: float,
    r: float,
    charfn: Callable[[np.ndarray], np.ndarray],
    alpha: float = 1.5,
    N: int = 2**12,
    eta: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute call prices over a strike grid via FFT.

    Parameters
    ----------
    S0 : float
        Spot.
    T : float
        Time to maturity (years).
    r : float
        Risk-free (cont. comp.).
    charfn : callable
        Function u -> φ_X(u) for X_T = log(S_T/S0) under Q.
    alpha : float, default 1.5
        Damping parameter (>0).
    N : int, default 4096
        FFT grid size (power of 2 recommended).
    eta : float, default 0.25
        Frequency domain grid spacing.

    Returns
    -------
    strikes : np.ndarray
    prices  : np.ndarray
    """
    assert alpha > 0.0, "alpha must be > 0"
    v = np.arange(N) * eta  # frequency grid
    # ψ(v) as per Carr–Madan
    numerator = charfn(v - 1j * (alpha + 1.0))
    denom = (alpha**2 + alpha - v**2) + 1j * (2.0 * alpha + 1.0) * v
    psi = np.exp(-r * T) * numerator / denom

    # Simpson weights
    w = np.ones(N)
    w[0] = w[-1] = 1.0 / 3.0
    w[1:-1:2] = 4.0 / 3.0
    w[2:-1:2] = 2.0 / 3.0
    psi_weighted = psi * w

    # FFT
    fft_vals = fft(psi_weighted).real  # real part after symmetry
    dk = 2.0 * np.pi / (N * eta)
    k = - (N // 2) * dk + dk * np.arange(N)  # log-strike grid centered near 0
    strikes = S0 * np.exp(k)

    # Undo damping and scale by π
    calls = np.exp(-alpha * k) * fft_vals / np.pi

    # For usability, return central half around ATM
    start = N // 4
    end = 3 * N // 4
    return strikes[start:end], calls[start:end]
