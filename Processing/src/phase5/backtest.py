from dataclasses import dataclass
import pandas as pd
import numpy as np

@dataclass
class BacktestConfig:
    slippage_bps: float = 1.0
    trade_on_open: bool = False

def backtest_signals(df: pd.DataFrame, cfg: BacktestConfig = BacktestConfig()) -> pd.DataFrame:
    out = df.copy().sort_values("date").reset_index(drop=True)
    if "underlying_close" in out.columns and out["underlying_close"].notna().any():
        out["ret"] = out["underlying_close"].pct_change().fillna(0.0)
    else:
        if "pcr_oi_chg1" in out.columns:
            out["ret"] = (-out["pcr_oi_chg1"]).fillna(0.0) * 0.01
        else:
            out["ret"] = 0.0
    out["position"] = out["signal"].shift(1).fillna(0.0)
    change = out["position"].diff().abs().fillna(out["position"].abs())
    tc = (cfg.slippage_bps/10000.0) * change
    out["strategy_ret"] = out["position"] * out["ret"] - tc
    out["cum_equity"] = (1.0 + out["strategy_ret"]).cumprod()
    out["cum_mkt"] = (1.0 + out["ret"]).cumprod()
    return out
