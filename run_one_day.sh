# 0) Env
cd Processing
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install feedparser beautifulsoup4 curl_cffi transformers torch
export PYTHONPATH="$PWD:$PYTHONPATH"

# 1) Get data for a target date (set as ENV; default: today IST)
DATE=$(TZ=Asia/Kolkata date +%F)

# 1a) News → CSV (always write CSV even if DB fails)
cd ../Data
# python3 historical_news_backfill.py "$DATE" "$DATE" --csv-out ./nifty_news_history.csv
# also dumps/updates per-day file via your collector
# python3 news_collector.py  # writes nifty_news_${DATE}.csv
# (If you prefer one source, keep either of the above.)

# 1b) Options → monthly → split per-day
# Option A: if you have vendor monthly file already:
#   python3 compute_iv_and_split.py --infile monthly_options_${DATE:0:7}.csv --outdir .
# Option B: (if your month ingestor is set up)
# python3 ingest_options_month.py --month "${DATE:0:7}" --out ./monthly_options_${DATE:0:7}.csv
# python3 compute_iv_and_split.py --infile ./monthly_options_${DATE:0:7}.csv --outdir .
./run_pipeline.sh
cd ../Processing

# 2) Build features
python3 -m src.phase5.cli build --data-dir ../Data --out phase5_features.csv

# 3) Train once (skip if model already exists)
[ -f models/phase5_baseline.joblib ] || \
python3 -m src.phase5.cli train --features phase5_features.csv \
  --model-out models/phase5_baseline.joblib \
  --metrics-out models/phase5_cv_metrics.csv

# 4) Score today (prints Prob(up))
python3 -m src.phase5.cli today --features phase5_features.csv --model models/phase5_baseline.joblib

# 5) (Optional) Backtest whole history with thresholds
python3 -m src.phase5.cli backtest --features phase5_features.csv --model models/phase5_baseline.joblib \
  --buy 0.60 --sell 0.40 --slippage 1.0 --out phase5_backtest.csv

# 6) (Optional) Phase-4 λ(t) sanity price on one option (after fixing run_price_with_sentiment)
# python - <<'PY'
# from src.phase4.run_price_with_sentiment import price_on_date
# print(price_on_date("phase5_features.csv", quote_date="$DATE", maturity_date="$DATE",
#                     S0=20000, r=0.06, K=20000, is_call=True,
#                     lam0=8.0, beta=0.5, kou_sigma=0.18, p_up=0.35, eta1=25.0, eta2=20.0))
# PY
