#!/usr/bin/env python3
"""
compute_iv_and_split.py

Reads a vendor monthly options CSV, computes Black–Scholes implied volatility (IV),
and writes per-day CSVs in the exact schema required by Phase-5:

Raw (per day):
  ExpiryDate, OptionType, StrikePrice, OpenInterest, ChangeInOI, LTP, IV

Aggregated (per day):
  ExpiryDate, OptionType, StrikePrice, OpenInterest, ChangeInOI, LTP, IV,
  OI_Daily_Change, IV_Daily_Change

Usage:
  python3 compute_iv_and_split.py --infile monthly_options.csv --outdir . --rf 0.06 --div 0.00

Notes:
- Date parsing assumes day-first formats like '26-Sep-25' or '26-09-2025'.
- OptionType is normalized to 'Call'/'Put' (CE/PE → Call/Put).
- Risk-free rate (rf) and dividend yield (div) are annualized; defaults are sensible but configurable.
"""

import argparse
from pathlib import Path
from datetime import datetime
import math
import sys

import pandas as pd

# ---------- Column alias maps (robust to vendor quirks) ----------
ALIASES = {
    "trade_date": [
        "Date", "date", "Trade Date", "TradeDate", "TradingDate",
        "TIMESTAMP", "timestamp", "datetime", "time"
    ],
    "expiry": ["Expiry", "Expiry Date", "expiry", "ExpiryDate", "EXPIRY_DT", "EXPIRY"],
    "type": ["Option type", "OptionType", "option_type", "type", "Right", "OPTION_TYP"],
    "strike": ["Strike Price", "StrikePrice", "strike", "Strike", "STRIKE_PR"],
    "oi": ["Open Int", "OpenInterest", "open_interest", "OI", "oi", "OPEN_INT"],
    "ch_oi": ["Change in OI", "ChangeInOI", "change_in_oi", "Chg in OI", "CHG_IN_OI", "Chg_OI"],
    "ltp": ["LTP", "ltp", "Last Traded Price", "Last Price", "Close", "CLOSE"],
    "under": ["Underlying Value", "underlying", "underlying_price", "Spot", "Underlying", "UNDERLYING_VALUE"],
    "symbol": ["Symbol", "symbol", "ticker", "Underlying Symbol", "SYMBOL"],
}

def pick(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None

# ---------- Black–Scholes & IV solver ----------
def _phi(x: float) -> float:
    """Standard normal CDF via error function (no SciPy)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def _bs_price(S, K, T, r, q, sigma, is_call: bool) -> float:
    """Black–Scholes price for European calls/puts with continuous dividend yield q."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        # At expiry, price ≈ intrinsic
        intrinsic = max(S - K, 0.0) if is_call else max(K - S, 0.0)
        return intrinsic
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    disc_r = math.exp(-r * T)
    disc_q = math.exp(-q * T)
    if is_call:
        return disc_q * S * _phi(d1) - disc_r * K * _phi(d2)
    else:
        return disc_r * K * _phi(-d2) - disc_q * S * _phi(-d1)

def implied_vol(price, S, K, T, r, q, is_call, tol=1e-6, max_iter=100, low=1e-6, high=5.0):
    """
    Simple bisection IV solver. Returns float('nan') if price is out of arbitrage bounds or no solution.
    """
    # Basic sanity & intrinsic bounds
    if any(not math.isfinite(x) for x in [price, S, K, T, r, q]) or S <= 0 or K <= 0 or T <= 0 or price <= 0:
        return float('nan')
    intrinsic = max(S - K, 0.0) if is_call else max(K - S, 0.0)
    if price < intrinsic - 1e-8:
        return float('nan')  # below intrinsic -> no solution
    # Upper bound (very rough): price <= S for call, <= K for put (discounting ignored deliberately conservative)
    if is_call and price > S:
        return float('nan')
    if (not is_call) and price > K:
        return float('nan')

    f_low = _bs_price(S, K, T, r, q, low, is_call) - price
    f_high = _bs_price(S, K, T, r, q, high, is_call) - price

    # If not bracketed, try expanding 'high'
    expand = 0
    while f_low * f_high > 0 and high < 10.0 and expand < 5:
        high *= 1.5
        f_high = _bs_price(S, K, T, r, q, high, is_call) - price
        expand += 1
    if f_low * f_high > 0:
        return float('nan')

    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        f_mid = _bs_price(S, K, T, r, q, mid, is_call) - price
        if abs(f_mid) < tol:
            return mid
        if f_low * f_mid <= 0:
            high = mid
            f_high = f_mid
        else:
            low = mid
            f_low = f_mid
    return 0.5 * (low + high)

# ---------- Main transform ----------
def normalize_monthly(infile: Path, rf: float, div: float) -> pd.DataFrame:
    df = pd.read_csv(infile)
    df.columns = df.columns.map(str).str.strip()

    # Resolve columns
    c_date  = pick(df.columns, ALIASES["trade_date"])
    c_exp   = pick(df.columns, ALIASES["expiry"])
    c_type  = pick(df.columns, ALIASES["type"])
    c_strk  = pick(df.columns, ALIASES["strike"])
    c_oi    = pick(df.columns, ALIASES["oi"])
    c_ch_oi = pick(df.columns, ALIASES["ch_oi"])
    c_ltp   = pick(df.columns, ALIASES["ltp"])
    c_under = pick(df.columns, ALIASES["under"])
    c_sym   = pick(df.columns, ALIASES["symbol"])

    missing = [name for name, col in [("trade_date", c_date), ("expiry", c_exp),
                                      ("type", c_type), ("strike", c_strk), ("oi", c_oi),
                                      ("ltp/close", c_ltp), ("underlying", c_under)]
               if col is None]
    if missing:
        print(f"[WARN] Missing columns: {missing}. IV will be NaN where price/underlying/time missing.")
    # Parse date and expiry
    # Day-first to handle '26-Sep-25'
    date_parsed = pd.to_datetime(df[c_date], dayfirst=True, errors="coerce") if c_date else pd.NaT
    expiry_parsed = pd.to_datetime(df[c_exp],  dayfirst=True, errors="coerce") if c_exp else pd.NaT
    df["__date__"] = date_parsed.dt.date.astype(str) if c_date else ""
    df["ExpiryDate"] = expiry_parsed.dt.strftime("%d-%b-%Y") if c_exp else ""

    # Normalize option type
    def norm_type(x):
        s = str(x).strip().upper()
        if s in {"CE", "CALL"}: return "Call"
        if s in {"PE", "PUT"}:  return "Put"
        return s.title() if s else ""
    df["OptionType"] = df[c_type].map(norm_type) if c_type else ""

    # Numerics
    df["StrikePrice"]  = pd.to_numeric(df[c_strk], errors="coerce") if c_strk else float("nan")
    df["OpenInterest"] = pd.to_numeric(df[c_oi],   errors="coerce") if c_oi else float("nan")
    df["ChangeInOI"]   = pd.to_numeric(df[c_ch_oi], errors="coerce") if c_ch_oi else 0.0
    df["LTP"]          = pd.to_numeric(df[c_ltp],  errors="coerce") if c_ltp else float("nan")
    df["UNDERLYING"]   = pd.to_numeric(df[c_under], errors="coerce") if c_under else float("nan")

    # Compute T in years ACT/365 (ensure non-negative)
    if c_date and c_exp:
        T_days = (expiry_parsed.dt.normalize() - date_parsed.dt.normalize()).dt.days
        df["__T__"] = (T_days.clip(lower=0) / 365.0)
    else:
        df["__T__"] = float("nan")

    # Compute IV row-wise (only when we have all needed inputs)
    iv_vals = []
    for S, K, T, price, ttype in zip(df["UNDERLYING"], df["StrikePrice"], df["__T__"], df["LTP"], df["OptionType"]):
        if not (math.isfinite(S) and math.isfinite(K) and math.isfinite(T) and math.isfinite(price)) or T <= 0 or S <= 0 or K <= 0 or price <= 0:
            iv_vals.append(float("nan"))
            continue
        is_call = (ttype == "Call")
        iv = implied_vol(price, S, K, T, rf, div, is_call)
        iv_vals.append(iv*100)
    df["IV"] = iv_vals

    # Keep only rows that have the essentials for output
    out = df[["__date__", "ExpiryDate", "OptionType", "StrikePrice", "OpenInterest", "ChangeInOI", "LTP", "IV"]].copy()
    out = out.dropna(subset=["__date__", "ExpiryDate", "OptionType", "StrikePrice", "OpenInterest"])
    # Replace NaN IV with 0.0 to satisfy downstream schema if needed
    out["IV"] = out["IV"].fillna(0.0)
    return out.rename(columns={"__date__": "date"})

def write_daily_files(df: pd.DataFrame, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    # Compute day-over-day deltas per key across all dates
    key = ["ExpiryDate", "OptionType", "StrikePrice"]
    df = df.sort_values(["date"] + key).reset_index(drop=True)
    prev_oi = df.groupby(key)["OpenInterest"].shift(1)
    prev_iv = df.groupby(key)["IV"].shift(1)
    df["OI_Daily_Change"] = (df["OpenInterest"] - prev_oi).fillna(0.0)
    df["IV_Daily_Change"] = (df["IV"] - prev_iv).fillna(0.0)

    # Write per-day files
    for day, g in df.groupby("date"):
        raw = g[["ExpiryDate", "OptionType", "StrikePrice", "OpenInterest", "ChangeInOI", "LTP", "IV"]]
        agg = g[["ExpiryDate", "OptionType", "StrikePrice", "OpenInterest", "ChangeInOI", "LTP", "IV",
                 "OI_Daily_Change", "IV_Daily_Change"]]
        (outdir / f"nifty_options_{day}.csv").write_text(raw.to_csv(index=False))
        (outdir / f"nifty_options_aggregated_{day}.csv").write_text(agg.to_csv(index=False))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", required=True, help="Path to monthly_options.csv")
    ap.add_argument("--outdir", required=True, help="Directory to write per-day files")
    ap.add_argument("--rf", type=float, default=0.06, help="Annualized risk-free rate (e.g., 0.06)")
    ap.add_argument("--div", type=float, default=0.00, help="Annualized dividend yield (e.g., 0.01)")
    args = ap.parse_args()

    infile = Path(args.infile)
    outdir = Path(args.outdir)

    df = normalize_monthly(infile, rf=args.rf, div=args.div)
    if df.empty:
        print("[ERROR] No usable rows parsed from the monthly file. Check column names and data.")
        sys.exit(1)
    write_daily_files(df, outdir)
    print(f"[OK] Wrote daily raw and aggregated files to: {outdir}")

if __name__ == "__main__":
    main()
