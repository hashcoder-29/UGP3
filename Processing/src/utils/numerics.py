"""
Numerical utilities: seeds, grids, stability helpers.
"""
from __future__ import annotations
import numpy as np
from typing import Tuple

def set_seed(seed: int = 7):
    np.random.seed(seed)

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def default_fft_density_grid(N: int = 2**12, x_span: float = 0.15) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build symmetric x-grid and matching u-grid for CF inversion.
    """
    x = np.linspace(-x_span, x_span, N)
    u_max = np.pi * (N - 1) / x_span
    u = np.linspace(-u_max, u_max, N)
    return x, u
