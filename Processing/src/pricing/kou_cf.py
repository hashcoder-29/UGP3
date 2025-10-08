"""
Kou (2002) double-exponential jump-diffusion characteristic function under Q.

phi_X(u) = exp( T * psi(u) )
psi(u) = i u (r - d - 0.5*sigma^2 - lambda * kappa) - 0.5*sigma^2 u^2
         + lambda * ( p*eta1/(eta1 - i u) + (1-p)*eta2/(eta2 + i u) - 1 )
kappa = E[e^Y] - 1 = p*eta1/(eta1 - 1) + (1-p)*eta2/(eta2 + 1) - 1
"""
from __future__ import annotations
from typing import Dict, Any
from src.utils.backend import xp, maybe_jit

@maybe_jit
def charfn_kou(u, T: float, params: Dict[str, Any]):
    """
    Risk-neutral CF of X_T = log(S_T / S_0) for Kou model.

    Parameters
    ----------
    u : array-like (real/complex)
    T : float  (years)
    params : dict with keys {"sigma","lambda","p","eta1","eta2","r","d"}

    Returns
    -------
    phi : same shape as u (complex)
    """
    sigma = float(params.get("sigma", 0.2))
    lam   = float(params.get("lambda", 0.0))
    p     = float(params.get("p", 0.5))
    eta1  = float(params.get("eta1", 20.0))
    eta2  = float(params.get("eta2", 25.0))
    r     = float(params.get("r", 0.0))
    d     = float(params.get("d", 0.0))

    # Feasibility (scalar checks are fine in JAX)
    if not (0.0 < p < 1.0) or (eta1 <= 1.0) or (eta2 <= 0.0) or (sigma < 0.0) or (lam < 0.0) or (T < 0.0):
        return xp.zeros_like(u, dtype=xp.complex128)

    u = xp.asarray(u, dtype=xp.complex128)
    i = 1j

    # Martingale correction for jumps
    kappa = p * (eta1 / (eta1 - 1.0)) + (1.0 - p) * (eta2 / (eta2 + 1.0)) - 1.0

    # Characteristic exponent
    psi = (i * u * (r - d - 0.5 * sigma * sigma - lam * kappa)
           - 0.5 * sigma * sigma * (u ** 2)
           + lam * (p * eta1 / (eta1 - i * u) + (1.0 - p) * eta2 / (eta2 + i * u) - 1.0))

    return xp.exp(T * psi)
