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
    out.index.name = "date"        # if you worked on a DateTimeIndex
    have_date_col = ("date" in out.columns) or ("Date" in out.columns)
    out = out.reset_index(drop=have_date_col)  # don't insert index as a column if date already exists
    if "Date" in out.columns and "date" not in out.columns:
        out = out.rename(columns={"Date": "date"})
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return out
  