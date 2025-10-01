# News-Driven Jump Dynamics in Option Pricing — Phase 3 Baseline (Kou + FFT)

This repo implements **Phase 3** of the project plan: a production-grade **Kou (2002)** double-exponential jump-diffusion model with a **Carr–Madan (1999) FFT** call pricer, plus calibration and simulation scaffolding. It also lays guardrails for **Phase 4** (news-driven Cox intensity).

> **Core outputs:**  
> - Vectorized **FFT pricer** for European calls  
> - **Kou characteristic function** under the risk-neutral measure  
> - **MLE calibration** to NIFTY 50 daily returns (10 years target; synthetic fallback)  
> - **Monte Carlo paths** (GBM + compound Poisson + double-exponential jumps) with hooks for **λ(t)**  
> - **Unit tests**: characteristic function sanity, FFT ↔ Black–Scholes limit, calibration validity

---

## 1) Theory Overview (Short)

### Kou jump-diffusion (risk-neutral)
Let \( S_t \) follow a diffusion with jumps (double-exponential log-jumps). For \( X_T=\ln(S_T/S_0) \),
\[
\phi_X(u)=\mathbb{E}^Q[e^{iuX_T}]=\exp\{\psi(u)T\},\quad
\psi(u)=iu(r-d-\tfrac12\sigma^2-\lambda\kappa)-\tfrac12\sigma^2u^2
+\lambda\big(\tfrac{p\eta_1}{\eta_1-iu}+\tfrac{(1-p)\eta_2}{\eta_2+iu}-1\big),
\]
where \( \kappa=\mathbb{E}[e^Y]-1=p\frac{\eta_1}{\eta_1-1}+(1-p)\frac{\eta_2}{\eta_2+1}-1 \).
Constraints: \( \eta_1>1,\ \eta_2>0,\ 0<p<1,\ \sigma\ge0,\ \lambda\ge0 \).

### Carr–Madan FFT (calls)
Define damped call \( c_\alpha(k)=e^{\alpha k} C(k) \) with \( k=\ln K,\ \alpha>0 \). Its FT is
\[
\psi_T(v)=e^{-rT}\frac{\phi_X\big(v-i(\alpha+1)\big)}{\alpha^2+\alpha-v^2+i(2\alpha+1)v}.
\]
Sampling \( v \) on an even grid and inverting by FFT yields call prices across strikes in one shot.

**References**  
- Kou, S. G. (2002), *Management Science*.  
- Carr, P., Madan, D. (1999), *Journal of Computational Finance*.

---

## 2) Symbols

| Symbol | Meaning |
|---|---|
| \( \sigma \) | Diffusive volatility (annualized) |
| \( \lambda \) | Jump intensity (jumps per year) |
| \( p \) | Prob. of upward jump (downward \(1-p\)) |
| \( \eta_1,\eta_2 \) | Double-exponential rates (up/down tails) |
| \( r \) | Risk-free rate (cont. comp.), India Gov’t yield |
| \( d \) | Dividend yield (0 for NIFTY index) |
| \( \alpha \) | Carr–Madan damping |
| \( N,\eta \) | FFT grid size and frequency spacing |

---

## 3) Install & Run

```bash
# Python 3.11+ recommended
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Run tests
pytest -q
