import numpy as np
from src.pricing.kou_cf import charfn_kou
from src.pricing.fft_pricer import price_calls_fft

def _bs_call(S, K, T, r, sigma):
    from math import log, sqrt, exp
    from mpmath import quad, erfc  # avoid pulling scipy.stats; keep lightweight
    # Closed-form Black-Scholes using standard normal CDF via error function
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    # Standard normal CDF
    Phi = lambda x: 0.5 * (1.0 + np.math.erf(x/np.sqrt(2.0)))
    return S*Phi(d1) - K*np.exp(-r*T)*Phi(d2)

def test_fft_matches_black_scholes_when_lambda_zero():
    S0, T, r, sigma = 100.0, 0.5, 0.03, 0.2
    params = dict(sigma=sigma, lambda=0.0, p=0.4, eta1=20.0, eta2=25.0, r=r, d=0.0)
    charfn = lambda u: charfn_kou(u, T, params)
    strikes, prices = price_calls_fft(S0, T, r, charfn, alpha=1.5, N=2**12, eta=0.25)
    # pick a few strikes around ATM
    for K in [80, 90, 100, 110, 120]:
        # interpolate FFT price
        price = np.interp(K, strikes, prices)
        bs = _bs_call(S0, K, T, r, sigma)
        assert np.isclose(price, bs, rtol=2e-3, atol=2e-3)  # ~0.2% tolerance
