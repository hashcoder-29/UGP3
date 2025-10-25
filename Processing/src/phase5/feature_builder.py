import re
from pathlib import Path
from typing import Optional, List
import pandas as pd
import numpy as np

def _find_files(base: Path, pattern: str) -> List[Path]:
    return sorted([p for p in base.glob(pattern) if p.is_file()])

def _parse_date_from_name(name: str) -> Optional[pd.Timestamp]:
    m = re.search(r"(20\d{2}-\d{2}-\d{2})", name)
    if m:
        try:
            return pd.to_datetime(m.group(1))
        except Exception:
            return None
    return None

def load_options_daily(base: Path) -> pd.DataFrame:
    files = _find_files(base, "nifty_options_aggregated_*.csv")
    if not files:
        files = _find_files(base, "nifty_options_*.csv")
    frames = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df["file_date"] = _parse_date_from_name(f.name)
            frames.append(df)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

def load_news_daily(base: Path) -> pd.DataFrame:
    files = _find_files(base, "nifty_news_*.csv")
    frames = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df["file_date"] = _parse_date_from_name(f.name)
            frames.append(df)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

def aggregate_news_sentiment(news_df: pd.DataFrame) -> pd.DataFrame:
    if news_df.empty:
        return pd.DataFrame(columns=["date","news_count","sent_pos_share","sent_neg_share","sent_neu_share","sent_score_mean","sent_score_std"])
    news_df = news_df.copy()
    if "file_date" in news_df.columns:
        news_df["date"] = pd.to_datetime(news_df["file_date"]).dt.date
    elif "date" in news_df.columns:
        news_df["date"] = pd.to_datetime(news_df["date"]).dt.date
    else:
        for col in ["published_at","publishedAt","time","timestamp"]:
            if col in news_df.columns:
                news_df["date"] = pd.to_datetime(news_df[col]).dt.date
                break
        if "date" not in news_df.columns:
            raise ValueError("No date column found in news data")
    news_df["sentiment_label"] = news_df.get("sentiment_label", pd.Series(["neutral"]*len(news_df)))
    news_df["sentiment_score"] = pd.to_numeric(news_df.get("sentiment_score", pd.Series([0.0]*len(news_df))), errors="coerce").fillna(0.0)
    g = news_df.groupby("date")
    agg = pd.DataFrame({
        "news_count": g.size(),
        "sent_pos_share": g.apply(lambda x: (x["sentiment_label"].str.lower()=="positive").mean()),
        "sent_neg_share": g.apply(lambda x: (x["sentiment_label"].str.lower()=="negative").mean()),
        "sent_neu_share": g.apply(lambda x: (x["sentiment_label"].str.lower()=="neutral").mean()),
        "sent_score_mean": g["sentiment_score"].mean(),
        "sent_score_std": g["sentiment_score"].std().fillna(0.0),
    }).reset_index()
    return agg

def aggregate_options_features(opt_df: pd.DataFrame) -> pd.DataFrame:
    if opt_df.empty:
        return pd.DataFrame(columns=["date","pcr_oi","pcr_volume","total_call_oi","total_put_oi","total_call_vol","total_put_vol","oi_change_sum","volume_sum","underlying_close"])
    df = opt_df.copy()
    date_col = None
    for c in ["date","Date","file_date","trade_date"]:
        if c in df.columns:
            date_col = c; break
    if date_col is None:
        raise ValueError("No date-like column in options data")
    df["date"] = pd.to_datetime(df[date_col]).dt.date

    def pick(cols):
        for c in cols:
            if c in df.columns:
                return c
        return None

    call_put_col = pick(["option_type","OptionType","type","Type"])
    oi_col = pick(["open_interest","OI","OpenInterest","openInterest"])
    vol_col = pick(["volume","Volume","totalTradedVolume","traded_volume"])
    oi_ch_col = pick(["change_in_oi","change_in_open_interest","Change_in_OI"])
    und_price_col = pick(["underlying","underlying_price","Underlying","Spot","close"])

    if call_put_col and call_put_col in df.columns:
        is_call = df[call_put_col].astype(str).str.upper().str.startswith("C")
        is_put  = df[call_put_col].astype(str).str.upper().str.startswith("P")
        calls = df[is_call].groupby("date").agg(
            total_call_oi=(oi_col,"sum") if oi_col in df.columns else ("date","size"),
            total_call_vol=(vol_col,"sum") if vol_col in df.columns else ("date","size"),
        )
        puts = df[is_put].groupby("date").agg(
            total_put_oi=(oi_col,"sum") if oi_col in df.columns else ("date","size"),
            total_put_vol=(vol_col,"sum") if vol_col in df.columns else ("date","size"),
        )
        base = calls.join(puts, how="outer").reset_index()
    else:
        base = df.groupby("date").agg(
            volume_sum=(vol_col,"sum") if vol_col in df.columns else ("date","size"),
            oi_change_sum=(oi_ch_col,"sum") if oi_ch_col in df.columns else ("date","size"),
        ).reset_index()
        base["total_call_oi"]=np.nan; base["total_put_oi"]=np.nan
        base["total_call_vol"]=np.nan; base["total_put_vol"]=np.nan

    base["pcr_oi"] = base.apply(lambda r: (r.get("total_put_oi")/r.get("total_call_oi")) if (pd.notna(r.get("total_put_oi")) and pd.notna(r.get("total_call_oi")) and r.get("total_call_oi") not in [0,np.nan]) else np.nan, axis=1)
    base["pcr_volume"] = base.apply(lambda r: (r.get("total_put_vol")/r.get("total_call_vol")) if (pd.notna(r.get("total_put_vol")) and pd.notna(r.get("total_call_vol")) and r.get("total_call_vol") not in [0,np.nan]) else np.nan, axis=1)

    if und_price_col and und_price_col in df.columns:
        und = df.groupby("date")[und_price_col].mean().rename("underlying_close").reset_index()
        base = base.merge(und, on="date", how="left")
    else:
        base["underlying_close"] = np.nan
    return base

def build_daily_features(data_dir: Path) -> pd.DataFrame:
    news_df = load_news_daily(data_dir)
    opt_df  = load_options_daily(data_dir)
    if news_df.empty:
        news_agg = pd.DataFrame(columns=["date","news_count","sent_pos_share","sent_neg_share","sent_neu_share","sent_score_mean","sent_score_std"])
    else:
        news_agg = aggregate_news_sentiment(news_df)
    if opt_df.empty:
        opt_agg = pd.DataFrame(columns=["date","pcr_oi","pcr_volume","underlying_close"])
    else:
        opt_agg  = aggregate_options_features(opt_df)
    if news_agg.empty and opt_agg.empty:
        return pd.DataFrame()
    if news_agg.empty:
        feats = opt_agg.copy()
    elif opt_agg.empty:
        feats = news_agg.copy()
    else:
        feats = pd.merge(opt_agg, news_agg, on="date", how="outer")
    feats = feats.sort_values("date").reset_index(drop=True)
    for c in ["pcr_oi","pcr_volume","underlying_close","news_count","sent_score_mean"]:
        if c in feats.columns:
            feats[c] = feats[c].fillna(method="ffill").fillna(method="bfill")
    for c in ["pcr_oi","pcr_volume","underlying_close","news_count","sent_score_mean","sent_pos_share","sent_neg_share"]:
        if c in feats.columns:
            feats[c+"_chg1"] = feats[c].diff()
    feats["date"] = pd.to_datetime(feats["date"])
    return feats
