import argparse
from pathlib import Path
import pandas as pd
from .feature_builder import build_daily_features
from .modeling import add_labels, train_tscv, save_model
from .signals import attach_signals, SignalConfig
from .backtest import backtest_signals, BacktestConfig
import joblib

def cmd_build(args):
    feats = build_daily_features(Path(args.data_dir))
    feats.to_csv(args.out, index=False)
    print(f"Wrote features to {args.out} with {len(feats)} rows.")

def cmd_train(args):
    df = pd.read_csv(args.features)
    df["date"] = pd.to_datetime(df["date"])
    df = add_labels(df)
    model, info = train_tscv(df)
    save_model(model, Path(args.model_out))
    pd.DataFrame(info["cv_metrics"]).to_csv(args.metrics_out, index=False)
    print("Model saved to", args.model_out, "| CV metrics ->", args.metrics_out)

def cmd_today(args):
    feats = pd.read_csv(args.features)
    feats["date"] = pd.to_datetime(feats["date"])
    latest = feats.sort_values("date").tail(1)
    model = joblib.load(args.model)
    X = latest[[c for c in latest.columns if c != "date"]].fillna(0.0)
    prob = float(model.predict_proba(X)[:,1][0])
    print(f"Prob(up) today = {prob:.4f}")

def cmd_backtest(args):
    df = pd.read_csv(args.features)
    df["date"] = pd.to_datetime(df["date"])
    model = joblib.load(args.model)
    X = df[[c for c in df.columns if c != "date"]].fillna(0.0)
    df["prob_up"] = model.predict_proba(X)[:,1]
    df = attach_signals(df, "prob_up", SignalConfig(prob_buy=args.buy, prob_sell=args.sell))
    bt = backtest_signals(df, BacktestConfig(slippage_bps=args.slippage))
    bt.to_csv(args.out, index=False)
    print("Backtest written to", args.out)

def main():
    ap = argparse.ArgumentParser(prog="phase5")
    sub = ap.add_subparsers()

    p_build = sub.add_parser("build", help="Build daily features from data directory")
    p_build.add_argument("--data-dir", required=True)
    p_build.add_argument("--out", default="phase5_features.csv")
    p_build.set_defaults(func=cmd_build)

    p_train = sub.add_parser("train", help="Train baseline classifier on features")
    p_train.add_argument("--features", required=True)
    p_train.add_argument("--model-out", default="models/phase5_baseline.joblib")
    p_train.add_argument("--metrics-out", default="models/phase5_cv_metrics.csv")
    p_train.set_defaults(func=cmd_train)

    p_today = sub.add_parser("today", help="Score latest row")
    p_today.add_argument("--features", required=True)
    p_today.add_argument("--model", required=True)
    p_today.set_defaults(func=cmd_today)

    p_bt = sub.add_parser("backtest", help="Backtest signals on full features")
    p_bt.add_argument("--features", required=True)
    p_bt.add_argument("--model", required=True)
    p_bt.add_argument("--buy", type=float, default=0.6)
    p_bt.add_argument("--sell", type=float, default=0.4)
    p_bt.add_argument("--slippage", type=float, default=1.0)
    p_bt.add_argument("--out", default="phase5_backtest.csv")
    p_bt.set_defaults(func=cmd_backtest)

    args = ap.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        ap.print_help()

if __name__ == "__main__":
    main()
