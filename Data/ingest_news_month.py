#!/usr/bin/env python3
import pandas as pd
from pathlib import Path
import argparse

ALIASES = [
    "published_date","published_at","publishedAt","pubDate",
    "timestamp","datetime","time","created_at","createdAt","date"
]

def main(infile, outdir, date_col=None, tz="Asia/Kolkata"):
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(infile)
    df.columns = [c.strip() for c in df.columns]

    col = date_col
    if not col:
        for c in ALIASES:
            if c in df.columns:
                col = c; break
    if not col:
        raise ValueError(f"Could not find a date column. Looked for: {ALIASES}. "
                         f"Pass --date-col <name> if your file uses a different name.")

    # Parse datetimes
    s = df[col]
    dt = pd.to_datetime(s, errors="coerce", utc=True)
    # If utc parse failed and looks like local naive, try without UTC then localize
    if dt.isna().all():
        dt = pd.to_datetime(s, errors="coerce")
        dt = dt.dt.tz_localize(tz, nonexistent='shift_forward', ambiguous='NaT')
    df["date"] = dt.dt.tz_convert(tz).dt.date.astype(str)

    keep = [c for c in ["date","title","source","link","summary","sentiment_label","sentiment_score"] if c in df.columns]
    for day, g in df.groupby("date"):
        g[keep].to_csv(out/f"nifty_news_{day}.csv", index=False)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--date-col", default=None, help="Explicit date/datetime column name (e.g., published_date)")
    ap.add_argument("--tz", default="Asia/Kolkata")
    args = ap.parse_args()
    main(args.infile, args.outdir, args.date_col, args.tz)
