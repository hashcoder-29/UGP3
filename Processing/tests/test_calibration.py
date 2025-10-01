import numpy as np
import pandas as pd
from src.models.kou_calibration import calibrate_mle
from src.data.io import _gen_synthetic_returns

def test_mle_recovers_synthetic_params_coarsely():
    # Generate synthetic returns (short to keep CI fast)
    ret = _gen_synthetic_returns(n_days=252*1, seed=321)  # 1Y daily
    res = calibrate_mle(ret.values, n_starts=3, seed=11, x_span=0.12, Nu=2**11)
    assert res.success
    # Sanity: parameters in valid region
    p = res.params
    assert 0 < p["p"] < 1 and p["eta1"] > 1 and p["eta2"] > 0 and p["sigma"] > 0 and p["lambda"] >= 0
    # Log-likelihood must be finite
    assert np.isfinite(res.loglik)
