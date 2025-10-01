#!/bin/bash
# run_backfill.sh - A helper script to run the historical news backfill.

# Exit immediately if a command exits with a non-zero status.
set -e

# Check if start and end dates are provided
if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Usage: ./run_backfill.sh <start_date> <end_date>"
  echo "Example: ./run_backfill.sh 2025-09-01 2025-09-25"
  exit 1
fi

echo "🚀 --- STARTING HISTORICAL NEWS BACKFILL from $1 to $2 --- 🚀"

conda run -n nifty_pipeline python historical_news_backfill.py $1 $2

echo "🎉 --- HISTORICAL BACKFILL COMPLETE --- 🎉"