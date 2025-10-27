import argparse
from pathlib import Path
import pandas as pd
from .feature_builder import build_daily_features
from .modeling import add_labels, train_tscv, save_model, FEATURE_COLS_DEFAULT
from .signals import attach_signals, SignalConfig
from .backtest import backtest_signals, BacktestConfig
import joblib

def _load_feature_cols(model_path: Path) -> list[str]:
    cols_file = Path(model_path).with_suffix(".cols.txt")
    if cols_file.exists():
        return [c for c in cols_file.read_text().splitlines() if c.strip()]
    # fallback if the sidecar is missing
    return FEATURE_COLS_DEFAULT

def _save_feature_cols(model_out: Path, cols: list[str]) -> None:
    Path(model_out).with_suffix(".cols.txt").write_text("\n".join(cols))

def cmd_build(args):
    feats = build_daily_features(Path(args.data_dir))
    feats.to_csv(args.out, index=False)
    print(f"Wrote features to {args.out} with {len(feats)} rows.")

def cmd_train(args):
    df = pd.read_csv(args.features)
    # Be explicit about date format to avoid pandas warnings
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    df = add_labels(df)

    # Train; info contains the exact feature columns used
    model, info = train_tscv(df)
    save_model(model, Path(args.model_out))

    # Persist train-time feature columns for inference/backtest
    used_cols = info.get("features_used", FEATURE_COLS_DEFAULT)
    _save_feature_cols(Path(args.model_out), used_cols)

    pd.DataFrame(info["cv_metrics"]).to_csv(args.metrics_out, index=False)
    print("Model saved to", args.model_out, "| CV metrics ->", args.metrics_out)

def cmd_today(args):
    feats = pd.read_csv(args.features)
    feats["date"] = pd.to_datetime(feats["date"], format="%Y-%m-%d", errors="coerce")
    latest = feats.sort_values("date").tail(1)

    model = joblib.load(args.model)
    feature_cols = _load_feature_cols(Path(args.model))

    # Use the exact same columns, same order; pass numpy to the pipeline
    X = latest.reindex(columns=feature_cols, fill_value=0.0).values
    prob = float(model.predict_proba(X)[:, 1][0])
    print(f"Prob(up) today = {prob:.4f}")

def cmd_backtest(args):
    df = pd.read_csv(args.features)
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")

    model = joblib.load(args.model)
    feature_cols = _load_feature_cols(Path(args.model))

    # Align features to what the scaler/model expects; pass numpy to the pipeline
    X = df.reindex(columns=feature_cols, fill_value=0.0).values
    df["prob_up"] = model.predict_proba(X)[:, 1]

    df = attach_signals(df, "prob_up", SignalConfig(prob_buy=args.buy, prob_sell=args.sell))
    bt = backtest_signals(df, BacktestConfig(slippage_bps=args.slippage))
    if "date" in bt.columns:
        bt = bt[["date"] + [c for c in bt.columns if c != "date"]]

    # --- ensure date is a visible column and first ---
    if "date" not in bt.columns:
        # if backtest_signals returned a frame indexed by date
        if bt.index.name in ("date", "Date") or hasattr(bt.index, "dtype"):
            bt = bt.reset_index()
            if "index" in bt.columns and "date" not in bt.columns:
                bt = bt.rename(columns={"index": "date"})
            if "Date" in bt.columns and "date" not in bt.columns:
                bt = bt.rename(columns={"Date": "date"})

    bt["date"] = pd.to_datetime(bt["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    bt = bt[["date"] + [c for c in bt.columns if c != "date"]]
    # -----------------------------------------------

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
