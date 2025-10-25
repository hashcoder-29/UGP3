# Phase 5 — Signal Layer, Modeling, and Backtest

This phase converts daily options+news data into features, trains a baseline classifier, generates signals, and backtests.

## Commands
```bash
# Build features (from dir where pipeline CSVs are saved)
python -m src.phase5.cli build --data-dir ./

# Train baseline model and save CV metrics
python -m src.phase5.cli train --features phase5_features.csv --model-out models/phase5_baseline.joblib

# Backtest full history using the saved model
python -m src.phase5.cli backtest --features phase5_features.csv --model models/phase5_baseline.joblib --out phase5_backtest.csv

# Get today's probability (most recent row)
python -m src.phase5.cli today --features phase5_features.csv --model models/phase5_baseline.joblib
