# Nifty 50 Financial & Sentiment Analysis Pipeline 📈

## Project Overview

This project is an end-to-end, daily data pipeline designed for students. It demonstrates key data engineering and AI concepts using only free, open-source tools that can be run on a personal computer.

The pipeline performs the following tasks every day:
1.  **Fetches** the Nifty 50 End-of-Day (EOD) options chain data from the NSE India website.
2.  **Collects** financial news articles related to the Nifty 50 and its constituent companies from various RSS feeds.
3.  **Analyzes** the sentiment of each news article using the FinBERT deep learning model.
4.  **Aggregates** the financial and sentiment data into a clean, single daily summary file.

This final dataset can be used for financial market analysis, time-series forecasting, or exploring the correlation between market sentiment and options data.

---

## Architecture Diagram

The pipeline follows a simple, sequential flow of data processing.

```
          +------------------------+
          |  options_scraper.py    |
          | (Scrapes NSE Website)  |
          +-----------+------------+
                      |
                      v
          +------------------------+
          | nifty_options_YYYY-MM-DD.csv |
          +------------------------+
                      |
                      v (Used in Step 4)
          +------------------------+      +------------------------+
          |    news_collector.py   |----->| nifty_news_YYYY-MM-DD.csv |
          | (Fetches RSS Feeds)    |      +-----------+------------+
          +------------------------+                  |
                                                      v
                                          +------------------------+
                                          | sentiment_analyzer.py  |
                                          | (Adds Sentiment Columns)|
                                          +-----------+------------+
                                                      |
                                                      v (File is overwritten)
                                          +------------------------+
                                          | nifty_news_YYYY-MM-DD.csv |
                                          |    (with sentiment)    |
                                          +-----------+------------+
                                                      |
                      +-------------------------------+
                      |
                      v
          +------------------------+
          |   data_aggregator.py   |
          | (Combines & Summarizes)|
          +-----------+------------+
                      |
                      v
          +------------------------+
          | daily_summary_YYYY-MM-DD.csv | (FINAL OUTPUT)
          +------------------------+

```

---

## Setup Instructions (Using Conda)

Follow these steps to set up your environment using Conda.

### 1. Prerequisites
* **Anaconda** or **Miniconda** installed on your system. You can download it from the [official Anaconda website](https://www.anaconda.com/products/distribution).

### 2. Create and Activate the Conda Environment
The recommended way to create the environment is by using the provided `environment.yml` file. This automatically installs all required packages and the correct Python version.

1.  **Create the environment from the file:**
    Open your terminal or Anaconda Prompt and navigate to the project directory. Then, run:
    ```bash
    conda env create -f environment.yml
    ```
    This command will create a new environment named `nifty_pipeline`.

2.  **Activate the new environment:**
    ```bash
    conda activate nifty_pipeline
    ```
    You will see `(nifty_pipeline)` prefixed to your command prompt, indicating the environment is active.

**Note:** The first time you run the sentiment analysis script, it will download the FinBERT model from Hugging Face (~420MB). This is a one-time download and is cached for future use.

### 3. Deactivating the Environment
When you are finished working on the project, you can deactivate the environment with:
```bash
conda deactivate
```

---

## How to Run the Pipeline

With your `nifty_pipeline` Conda environment active, you can run the entire pipeline with a single command.

* **On macOS/Linux:**
    ```bash
    chmod +x run_pipeline.sh  # Make the script executable (only need to do this once)
    ./run_pipeline.sh
    ```

* **On Windows:**
    ```bash
    .\run_pipeline.bat
    ```

The script will execute each Python file in order, and you will see progress messages in your terminal. At the end, you will have three new CSV files for the current day.

---

## File Descriptions

### Environment Files
* `environment.yml`: The definition file for creating the Conda environment with all necessary dependencies.
* `requirements.txt`: A list of Python packages, primarily for reference or for users who prefer `pip`.

### Scripts (`.py`)
* `options_scraper.py`: Connects to the NSE website and downloads the raw Nifty 50 options chain data.
* `news_collector.py`: Scrapes the list of Nifty 50 companies and then scours financial news RSS feeds for relevant articles.
* `sentiment_analyzer.py`: Loads the FinBERT model to perform financial sentiment analysis on the news.
* `data_aggregator.py`: The final step. It reads the options and news data, calculates summary statistics, and creates the final daily dataset.

### Orchestration (`.sh`, `.bat`)
* `run_pipeline.sh` / `run_pipeline.bat`: Master scripts that run the entire pipeline in the correct sequence.

### Output Files (`.csv`)
* `nifty_options_YYYY-MM-DD.csv`: Raw EOD options chain data for Nifty 50.
* `nifty_news_YYYY-MM-DD.csv`: Collected news articles, enriched with sentiment labels and scores.
* `daily_summary_YYYY-MM-DD.csv`: The final, aggregated output containing key metrics for the day.