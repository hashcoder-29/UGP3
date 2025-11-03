#!/usr/bin/env python3
import pandas as pd, argparse
from pathlib import Path

ALIASES = {
    "date": ["Date","Trade Date","date","trade_date","timestamp","datetime","time"],
    "type": ["Option type","option_type","type","right"],
    "oi":   ["Open Int","open_interest","OI","oi","OpenInterest"],
    "vol":  ["Volume","volume","Traded Qty","traded_qty","qty","Quantity"],
    "under":["Underlying Value","underlying","underlying_price","Spot","close","Underlying"],
    "sym":  ["Symbol","symbol","ticker","underlying_symbol"]
}

def pick(cols, names):
    for n in names:
        if n in cols: return n
    return None

def main(infile, outdir, symbol_filter="NIFTY"):
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(infile)
    df.columns = df.columns.str.strip()  # remove trailing spaces
    df = df.replace({'-': None})

    # Resolve columns
    c_date  = pick(df.columns, ALIASES["date"])
    c_type  = pick(df.columns, ALIASES["type"])
    c_oi    = pick(df.columns, ALIASES["oi"])
    c_vol   = pick(df.columns, ALIASES["vol"])
    c_under = pick(df.columns, ALIASES["under"])
    c_sym   = pick(df.columns, ALIASES["sym"])

    if not c_date or not c_type or not c_oi:
        missing = [k for k,v in [("date",c_date),("type",c_type),("oi",c_oi)] if v is None]
        raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")

    # Parse date like '26-Sep-25'
    d = pd.to_datetime(df[c_date], dayfirst=True, errors="coerce")
    df["date"] = d.dt.date.astype(str)

    # Normalize fields
    df["option_type"] = (df[c_type].astype(str)
                                  .str.upper()
                                  .str.replace("CALL","CE", regex=False)
                                  .str.replace("PUT","PE",  regex=False))
    df["open_interest"] = pd.to_numeric(df[c_oi], errors="coerce")
    if c_vol:   df["volume"] = pd.to_numeric(df[c_vol], errors="coerce")
    if c_under: df["underlying"] = pd.to_numeric(df[c_under], errors="coerce")

    # Filter by underlying symbol
    if c_sym and symbol_filter:
        df = df[df[c_sym].astype(str).str.upper().str.contains(symbol_filter)]

    # Write RAW daily
    cols = ["date","option_type","open_interest"] + (["volume"] if "volume" in df.columns else []) + (["underlying"] if "underlying" in df.columns else [])
    for day, g in df.groupby("date"):
        g[cols].to_csv(out/f"nifty_options_{day}.csv", index=False)

    # Write AGGREGATED daily
