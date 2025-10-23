
# Phase 4 — NLP-driven Cox intensity extension (implementation scaffold)

- `cox_kou_mc.py` — simulate Kou with sentiment-driven Cox intensity; Monte Carlo pricing.
- `calibration_stage2.py` — stage-2 fit of intensity dynamics parameters vs market option prices.
- `sentiment_signal.py` — tiny helper to create a callable sentiment function.

Usage sketch:
```python
from src.phase4.cox_kou_mc import DiffusionParams, KouJumpParams, IntensityDynamics, price_european_mc
from src.phase4.sentiment_signal import make_step_fn

diff = DiffusionParams(mu=0.0, sigma=0.2)
jump = KouJumpParams(lam=1.0, p_up=0.3, eta1=15.0, eta2=8.0)
inten = IntensityDynamics(kappa=1.0, a=0.5, b=2.0, nu=0.2, lambda0=0.5)
senti = make_step_fn([0.0, 0.5, 1.0], [0.0, 0.5, 0.2])

price = price_european_mc(S0=100, K=100, T=1.0, r=0.01, diffusion=diff, jump=jump, intensity=inten, sentiment_fn=senti)
print(price)
```
