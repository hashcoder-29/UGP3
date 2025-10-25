from dataclasses import dataclass
import pandas as pd

@dataclass
class SignalConfig:
    prob_buy: float = 0.6
    prob_sell: float = 0.4
    max_position: int = 1

def make_signal(prob_up: float, cfg: SignalConfig) -> int:
    if prob_up >= cfg.prob_buy:
        return +cfg.max_position
    if prob_up <= cfg.prob_sell:
        return -cfg.max_position
    return 0

def attach_signals(df: pd.DataFrame, prob_col: str = "prob_up", cfg: SignalConfig = SignalConfig()) -> pd.DataFrame:
    out = df.copy()
    out["signal"] = out[prob_col].apply(lambda p: make_signal(p, cfg))
    return out
