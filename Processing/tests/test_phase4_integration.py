# tests/test_phase4_integration.py
import numpy as np
from src.phase4.sentiment_signal import build_lambda_curve_from_features
import pandas as pd

def test_lambda_positive_monotone():
    df = pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=10, freq="B"),
        "sent_score_mean": np.linspace(-2, 2, 10)
    })
    t, lam, f = build_lambda_curve_from_features(df, lam0=8.0, beta=0.5, lam_min=0.5, lam_max=40.0)
    assert (lam > 0).all()
    assert lam[-1] > lam[0]
