# historical_news_backfill.py
import os
import pandas as pd
import requests
from datetime import datetime, timedelta
import time
import argparse
import logging
from transformers import pipeline
import torch
from db_utils import get_db_engine

# ##############################################################################
# ## PASTE YOUR NEWS_API_KEY HERE                                             ##
NEWS_API_KEY = "6532d685925c4cb0bc86026474157de2"  # From https://newsapi.org/
# ##############################################################################

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
import time
from curl_cffi import requests as curl_requests

NSE_PAGE_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept-Language": "en-US,en;q=0.9",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36",
}

# Specialized headers for the subsequent API calls
NSE_API_HEADERS = {
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept-Language": "en-US,en;q=0.9",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36",
    # This Referer is critical. It makes the request look like it's coming from the option chain page.
    "Referer": "https://www.nseindia.com/option-chain",
    "X-Requested-With": "XMLHttpRequest",
}


def get_nse_session():
    """Initialize a single, reusable NSE session using curl_cffi."""
    try:
        session = curl_requests.Session()
        # ## --- FIX: Use the specific Page Headers for this initial visit --- ##
        r = session.get(
            "https://www.nseindia.com/option-chain", 
            headers=NSE_PAGE_HEADERS, 
            timeout=15, 
            impersonate="chrome110"
        )
        r.raise_for_status()
        logging.info("NSE session initialized successfully.")
        return session
    except Exception as e:
        logging.warning(f"Failed to initialize NSE session: {e}")
        return None
def _load_seen_links_from_csv(csv_path: str) -> set:
    seen = set()
    if not csv_path or not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return seen
    try:
        # read just the 'link' column fast; handles large files in chunks if needed later
        for chunk in pd.read_csv(csv_path, usecols=['link'], chunksize=100_000):
            seen.update(chunk['link'].dropna().astype(str).tolist())
    except Exception as e:
        logging.warning(f"Could not read existing CSV for dedupe: {e}")
    return seen

def get_nifty50_constituents(session):
    """Fetches Nifty 50 constituents using the provided curl_cffi session."""
    url = "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%2050"
    try:
        # ## --- FIX: Use the specific API Headers for this call --- ##
        response = session.get(url, headers=NSE_API_HEADERS, timeout=10)
        response.raise_for_status()
        data = response.json()["data"]
        constituents = {item["symbol"]: item["identifier"] for item in data}
        logging.info(f"Successfully fetched {len(constituents)} Nifty 50 constituents.")
        return constituents
    except Exception as e:
        logging.error(f"Error fetching Nifty 50 constituents: {e}. Using fallback list.")
        return {"RELIANCE": "Reliance Industries Ltd.", "TCS": "Tata Consultancy Services Ltd."}


def fetch_historical_newsapi(company_names, date_str):
    """ Fetches historical news for a specific day from NewsAPI. """
    if not NEWS_API_KEY:
        logging.warning("NewsAPI key not provided. Skipping.")
        return []
    
    articles = []
    logging.info(f"Fetching historical news from NewsAPI for {date_str}...")
    
    # Create a combined query for all company names to be efficient
    query = " OR ".join([f'"{name}"' for name in company_names]) + ' OR "Nifty 50"'
    
    try:
        url = (f"https://newsapi.org/v2/everything?q={query}&language=en"
               f"&from={date_str}&to={date_str}"
               f"&sortBy=relevancy&apiKey={NEWS_API_KEY}")
        
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        if data.get('status') == 'error':
            raise Exception(data.get('message'))
        
        for article in data.get('articles', []):
            link = article.get('url')
            if link:
                articles.append({
                    'source': f"NewsAPI - {article.get('source', {}).get('name')}",
                    'title': article.get('title'),
                    'summary': article.get('description'),
                    'link': link,
                    'published_date': article.get('publishedAt')
                })
    except Exception as e:
        logging.error(f"Could not fetch from NewsAPI for {date_str}: {e}")
        
    logging.info(f"Found {len(articles)} articles from NewsAPI for {date_str}.")
    return articles

def analyze_sentiment(df):
    """ Analyzes sentiment of news articles in a DataFrame. """
    if df.empty:
        return df
    
    logging.info(f"Starting sentiment analysis for {len(df)} articles...")
    try:
        device = 0 if torch.cuda.is_available() else -1
        sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert", device=device)
        
        # Combine title and summary for analysis
        texts_to_analyze = [f"{str(row['title'])}. {str(row['summary'])}" for _, row in df.iterrows()]
        
        # Process in batches for efficiency
        results = sentiment_pipeline(texts_to_analyze, truncation=True, max_length=512)
        
        df['sentiment_label'] = [res['label'] for res in results]
        df['sentiment_score'] = [res['score'] for res in results]
        logging.info("Sentiment analysis complete.")
    except Exception as e:
        logging.error(f"Error during sentiment analysis: {e}")
        df['sentiment_label'] = 'error'
        df['sentiment_score'] = 0.0
        
    return df

def _append_to_csv(df: pd.DataFrame, csv_path: str):
    """Append df to csv_path, creating file with header if missing/empty."""
    try:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True) if os.path.dirname(csv_path) else None
        write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
        df.to_csv(csv_path, mode='a', header=write_header, index=False, encoding='utf-8')
        logging.info(f"Appended {len(df)} rows to CSV: {csv_path}")
    except Exception as e:
        logging.error(f"Failed writing to CSV {csv_path}: {e}")

def run_backfill(start_date, end_date, csv_out: str | None = None):
    """Backfill: ALWAYS append to CSV (if provided), and also try DB if available."""
    engine = get_db_engine()
    db_enabled = engine is not None
    if not db_enabled:
        logging.error("DB engine not available. Will skip DB writes but continue CSV.")

    master_table_name = "historical_news"

    # NSE list (your session code stays the same)
    nse_session = get_nse_session()
    if nse_session:
        constituents = get_nifty50_constituents(nse_session)  # dict
        # Use full names for NewsAPI query
        nifty50_companies = list(constituents.values())
        logging.info(f"Using {len(nifty50_companies)} NIFTY 50 names from NSE API.")
    else:
        logging.error("Could not create NSE session. Using fallback for constituents.")
        nifty50_companies = [
            "Reliance Industries", "TCS", "HDFC Bank", "Infosys", "ICICI Bank",
            "Hindustan Unilever", "State Bank of India", "Bajaj Finance", "Bharti Airtel", "ITC"
        ]

    # Optional: your manual override — remove this if not intended
    nifty50_companies = ["Nifty 50", "NSE", "Sensex", "Indian market", "RBI"]

    # DB de-dupe set
    if db_enabled:
        try:
            existing_links_df = pd.read_sql_query(f'SELECT link FROM {master_table_name}', engine)
            seen_db = set(existing_links_df['link'].astype(str))
            logging.info(f"Loaded {len(seen_db)} existing article links from DB.")
        except Exception as e:
            logging.warning(f"Could not load links from DB: {e}")
            seen_db = set()
    else:
        seen_db = set()

    # CSV de-dupe set (so we don't re-write to CSV on reruns)
    seen_csv = _load_seen_links_from_csv(csv_out) if csv_out else set()
    if csv_out:
        logging.info(f"Loaded {len(seen_csv)} existing links from CSV for dedupe.")

    current_date = start_date
    while current_date <= end_date:
        date_str = current_date.strftime('%Y-%m-%d')

        # 1) Fetch
        daily_articles = fetch_historical_newsapi(nifty50_companies, date_str)
        if not daily_articles:
            logging.info(f"No articles found for {date_str}.")
            current_date += timedelta(days=1)
            time.sleep(2)
            continue

        daily_df = pd.DataFrame(daily_articles)

        # ---------- CSV path: ALWAYS attempt to write ----------
        if csv_out:
            # de-dupe vs CSV to avoid duplicates on reruns
            csv_new_df = daily_df[~daily_df['link'].astype(str).isin(seen_csv)]
            if not csv_new_df.empty:
                analyzed_csv_df = analyze_sentiment(csv_new_df.copy())
                cols = ['source','title','summary','link','published_date','sentiment_label','sentiment_score']
                analyzed_csv_df = analyzed_csv_df[[c for c in cols if c in analyzed_csv_df.columns]]
                _append_to_csv(analyzed_csv_df, csv_out)
                # update in-memory CSV seen set
                seen_csv.update(analyzed_csv_df['link'].astype(str))
            else:
                logging.info(f"CSV: No new links for {date_str} after CSV dedupe.")

        # ---------- DB path: best-effort, independent from CSV ----------
        if db_enabled:
            db_new_df = daily_df[~daily_df['link'].astype(str).isin(seen_db)]
            if db_new_df.empty:
                logging.info(f"DB: No new links for {date_str} after DB dedupe.")
            else:
                analyzed_db_df = analyze_sentiment(db_new_df.copy())
                cols = ['source','title','summary','link','published_date','sentiment_label','sentiment_score']
                analyzed_db_df = analyzed_db_df[[c for c in cols if c in analyzed_db_df.columns]]
                try:
                    analyzed_db_df.to_sql(master_table_name, engine, if_exists='append', index=False)
                    seen_db.update(analyzed_db_df['link'].astype(str))
                    logging.info(f"DB: Saved {len(analyzed_db_df)} new rows for {date_str}.")
                except Exception as e:
                    logging.error(f"DB write failed for {date_str}: {e}")

        current_date += timedelta(days=1)
        time.sleep(2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch historical news for Nifty 50 companies.")
    parser.add_argument("start_date", type=str, help="Start date in YYYY-MM-DD format.")
    parser.add_argument("end_date", type=str, help="End date in YYYY-MM-DD format.")
    parser.add_argument("--csv-out", type=str, default=None,
                        help="Optional path to append a CSV copy of the results (e.g., ./Data/historical_news.csv).")
    args = parser.parse_args()

    start = datetime.strptime(args.start_date, "%Y-%m-%d")
    end = datetime.strptime(args.end_date, "%Y-%m-%d")

    run_backfill(start, end, csv_out=args.csv_out)
    logging.info("Historical backfill process complete.")
