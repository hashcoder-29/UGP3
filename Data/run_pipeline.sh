#!/bin/bash
# run_pipeline.sh - Executes the entire data pipeline in sequence.

# Exit immediately if a command exits with a non-zero status.
set -e

echo "🚀 --- STARTING DAILY DATA PIPELINE --- 🚀"

# Step 1: Ingest Today's Raw Options Data
echo "
[Step 1/6] Running Options Scraper..."
conda run -n nifty_pipeline python options_scraper.py
echo "✅ Options Scraper Done."

# Step 2: Aggregate Today's Data with Previous Day's Data
echo "
[Step 2/6] Running Daily Options Aggregator..."
conda run -n nifty_pipeline python options_aggregator.py
echo "✅ Daily Options Aggregator Done."

# Step 3: Ingest News Data
echo "
[Step 3/6] Running News Collector..."
conda run -n nifty_pipeline python news_collector.py
echo "✅ News Collector Done."

# Step 4: Analyze Sentiment
echo "
[Step 4/6] Running Sentiment Analyzer..."
conda run -n nifty_pipeline python sentiment_analyzer.py
echo "✅ Sentiment Analyzer Done."

# Step 5: Aggregate Final Summary Data
echo "
[Step 5/6] Running Data Aggregator..."
conda run -n nifty_pipeline python data_aggregator.py
echo "✅ Data Aggregator Done."

# --- NEW STEP ---
# Step 6: Archive Options Data and Cleanup Daily Files
echo "
[Step 6/6] Running Options Archiver and Cleanup..."
conda run -n nifty_pipeline python options_archiver.py
echo "✅ Options Archiving and Cleanup Done."

echo "
🎉 --- PIPELINE EXECUTION COMPLETE --- 🎉"