"""
Kou (2002) double-exponential jump-diffusion characteristic function (risk-neutral).

References:
- Kou, S. G. (2002). "A Jump-Diffusion Model for Option Pricing". Management Science.
- Carr, P., Madan, D. (1999). "Option valuation using the fast Fourier transform".

We model X_T = log(S_T / S_0). Under Q:
  φ_X(u) = E^Q[exp(i u X_T)] = exp{ ψ(u) * T }
with
  ψ(u) = i u (r - d - 0.5 σ^2 - λ κ) - 0.5 σ^2 u^2
         + λ [ p η1 / (η1 - i u) + (1-p) η2 / (η2 + i u) - 1 ]
where κ = E[V - 1] = E[e^Y] - 1,
      Y ~ double-exponential: P(Y≥0)=p with density η1 e^{-η1 y} for y≥0,
                              P(Y<0)=1-p with density η2 e^{η2 y} for y<0.
Require η1>1 (finite upward mean), η2>0, 0<p<1, σ≥0, λ≥0.
"""
from __future__ import annotations
import numpy as np
from typing import Dict

def charfn_kou(u: np.ndarray, T: float, params: Dict[str, float]) -> np.ndarray:
    """
    Vectorized Kou characteristic function φ_X(u) for X_T = log(S_T/S_0).

    Parameters
    ----------
    u : array_like
        Fourier frequencies (float or complex), broadcastable.
    T : float
        Time to maturity in years.
    params : dict
        Expected keys: 'sigma','lambda','p','eta1','eta2','r','d' (d optional -> 0.0)

    Returns
    -------
    np.ndarray (complex)
        φ_X(u) evaluated at input u.
    """
    sigma = float(params["sigma"])
    lam = float(params["lambda"])
    p = float(params["p"])
    eta1 = float(params["eta1"])
    eta2 = float(params["eta2"])
    r = float(params.get("r", 0.0))
    d = float(params.get("d", 0.0))

    # Safety/feasibility checks (soft; allow optimizer to explore but avoid NaNs)
    if not (0.0 < p < 1.0) or eta1 <= 1.0 or eta2 <= 0.0 or sigma < 0.0 or lam < 0.0:
        # Return zeros to penalize invalid regions in calibration (log-likelihood -> -inf)
        u = np.asarray(u, dtype=complex)
        return np.zeros_like(u, dtype=complex)

    u = np.asarray(u, dtype=complex)
    iu = 1j * u

    # E[V] = E[e^Y] = p * η1/(η1-1) + (1-p) * η2/(η2+1)
    E_V = p * (eta1 / (eta1 - 1.0)) + (1.0 - p) * (eta2 / (eta2 + 1.0))
    kappa = E_V - 1.0

    drift = iu * (r - d - 0.5 * sigma**2 - lam * kappa)
    diffusion = -0.5 * (sigma**2) * (u**2)
    jumps = lam * (p * eta1 / (eta1 - iu) + (1.0 - p) * eta2 / (eta2 + iu) - 1.0)
    psi = drift + diffusion + jumps

    return np.exp(psi * T)
