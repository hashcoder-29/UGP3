#!/usr/bin/env bash
set -euo pipefail

# cd to repo root (Processing/)
cd "$(dirname "${BASH_SOURCE[0]}")"/..

# make sure Python can see src/
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

python -m src.phase5.cli build --data-dir ./
python -m src.phase5.cli train --features phase5_features.csv --model-out models/phase5_baseline.joblib
python -m src.phase5.cli backtest --features phase5_features.csv --model models/phase5_baseline.joblib --out phase5_backtest.csv

echo "Artifacts: phase5_features.csv, models/phase5_baseline.joblib, phase5_backtest.csv"
