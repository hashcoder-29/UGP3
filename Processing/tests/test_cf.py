import numpy as np
from src.pricing.kou_cf import charfn_kou

def test_cf_at_zero_is_one():
    params = dict(sigma=0.2, lambda=0.0, p=0.4, eta1=20.0, eta2=25.0, r=0.05, d=0.0)
    phi0 = charfn_kou(np.array([0.0]), 1.0, params)[0]
    assert np.isclose(phi0, 1.0 + 0.0j)

def test_cf_reduces_to_gbm_when_no_jumps():
    params = dict(sigma=0.2, lambda=0.0, p=0.4, eta1=20.0, eta2=25.0, r=0.03, d=0.0)
    T = 0.5
    u = np.linspace(-50, 50, 1001)
    phi = charfn_kou(u, T, params)
    # GBM φ: exp(iu (r - d - 0.5σ^2)T - 0.5 σ^2 u^2 T)
    sigma = params["sigma"]; r = params["r"]; d = params["d"]
    gbm_phi = np.exp(1j*u*(r-d-0.5*sigma**2)*T - 0.5*sigma**2*(u**2)*T)
    assert np.allclose(phi, gbm_phi, atol=1e-10)
