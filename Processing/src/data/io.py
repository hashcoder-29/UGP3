"""
Data I/O adapters:
- Try to read NIFTY prices/returns and yields from Postgres (SQLAlchemy/psycopg2).
- If unavailable, generate (and cache) small synthetic CSVs in ./data for tests/local use.

Env for DB (optional):
  DB_URI=postgresql+psycopg2://user:pass@host:port/dbname
Tables assumed (adapt to your schema if different):
  - historical_options (not used in Phase 3 baseline)
  - historical_news     (not used in Phase 3 baseline)
  - nifty_prices (date, close)               [optional]
  - indian_yields (date, tenor, yield_annual)[optional]

Fallback:
  - synthetic_returns.csv (daily log-returns)
  - synthetic_yield.csv   (date, rf)
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")

def _ensure_dir():
    os.makedirs(DATA_DIR, exist_ok=True)

def _gen_synthetic_returns(n_days: int = 252*2, seed: int = 123) -> pd.Series:
    """Generate small Kou-like synthetic daily log-returns for tests."""
    rng = np.random.default_rng(seed)
    mu = 0.10
    sigma = 0.18
    lam = 8.0
    p = 0.3
    eta1 = 25.0
    eta2 = 20.0
    dt = 1.0/252.0
    x = np.empty(n_days)
    for t in range(n_days):
        n = rng.poisson(lam*dt)
        y = 0.0
        if n > 0:
            u = rng.random(n)
            y_up = rng.exponential(1.0/eta1, (u< p).sum()).sum()
            y_dn = -rng.exponential(1.0/eta2, (u>=p).sum()).sum()
            y = y_up + y_dn
        diff = (mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*rng.standard_normal()
        x[t] = diff + y
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=n_days, freq="B")
    return pd.Series(x, index=dates, name="log_return")

def _gen_synthetic_rf(n_days: int = 252*2) -> pd.Series:
    """Flat 5.5% annualized rf, daily index."""
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=n_days, freq="B")
    rf = pd.Series(0.055, index=dates, name="rf")
    return rf

def load_returns_and_rf(use_db: bool = True, years: int = 10) -> Tuple[pd.Series, pd.Series]:
    """
    Try DB first. If fails, use/generate synthetic CSVs in ./data.
    Returns daily log-returns Series and daily rf Series (aligned index if possible).
    """
    _ensure_dir()
    if use_db and "DB_URI" in os.environ:
        try:
            import sqlalchemy as sa
            eng = sa.create_engine(os.environ["DB_URI"])
            # Pull last `years` of NIFTY close prices (assume table exists)
            q_prices = f"""
                SELECT date, close FROM nifty_prices
                WHERE date >= (CURRENT_DATE - interval '{years} years')
                ORDER BY date;
            """
            dfp = pd.read_sql(q_prices, eng, parse_dates=["date"]).set_index("date")
            # Construct log-returns
            ret = np.log(dfp["close"]).diff().dropna()
            # Pull yields; prefer 91-day T-bill; fall back to 1Y if needed
            q_y = f"""
                SELECT date, yield_annual FROM indian_yields
                WHERE tenor IN ('91D','3M','1Y')
                  AND date >= (CURRENT_DATE - interval '{years} years')
                ORDER BY date;
            """
            dfy = pd.read_sql(q_y, eng, parse_dates=["date"]).set_index("date")
            rf = dfy.groupby(level=0).first()["yield_annual"].reindex(ret.index).fillna(method="ffill")
            return ret, rf
        except Exception as e:
            # Fall back to local
            pass

    # Local synthetic fallback
    ret_path = os.path.join(DATA_DIR, "synthetic_returns.csv")
    rf_path = os.path.join(DATA_DIR, "synthetic_yield.csv")
    if not os.path.exists(ret_path):
        _gen_synthetic_returns().to_csv(ret_path, header=True)
    if not os.path.exists(rf_path):
        _gen_synthetic_rf().to_csv(rf_path, header=True)
    returns = pd.read_csv(ret_path, parse_dates=[0], index_col=0)["log_return"]
    rf = pd.read_csv(rf_path, parse_dates=[0], index_col=0)["rf"]
    # Align
    idx = returns.index.intersection(rf.index)
    returns = returns.loc[idx]
    rf = rf.loc[idx]
    return returns, rf
