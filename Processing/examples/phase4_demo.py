
from src.phase4.cox_kou_mc import DiffusionParams, KouJumpParams, IntensityDynamics, price_european_mc
from src.phase4.sentiment_signal import make_step_fn

def main():
    # Baseline diffusion and jump params (illustrative values)
    diff = DiffusionParams(mu=0.0, sigma=0.2)
    jump = KouJumpParams(lam=1.0, p_up=0.3, eta1=15.0, eta2=8.0)
    inten = IntensityDynamics(kappa=1.5, a=0.5, b=2.0, nu=0.2, lambda0=0.5)

    # sentiment: neutral then elevated around mid horizon
    tgrid = [0.0, 0.4, 0.7, 1.0]
    values = [0.0, 0.0, 0.8, 0.1]
    senti = make_step_fn(tgrid, values)

    price = price_european_mc(S0=100.0, K=100.0, T=1.0, r=0.01, diffusion=diff, jump=jump, intensity=inten,
                              n_paths=20000, n_steps=252, call=True, sentiment_fn=senti, seed=123)
    print(f"MC European Call Price (Phase 4 model): {price:.4f}")

if __name__ == "__main__":
    main()
