# historical_news_backfill.py
import pandas as pd
import requests
from datetime import datetime, timedelta
import time
import argparse
import logging
from transformers import pipeline
import torch

# ##############################################################################
# ## PASTE YOUR NEWS_API_KEY HERE                                             ##
NEWS_API_KEY = "6532d685925c4cb0bc86026474157de2"  # From https://newsapi.org/
# ##############################################################################

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
               f"&from={date_str}&to={date_str}"  # Search for a single day
               f"&sortBy=relevancy&apiKey={NEWS_API_KEY}")
        
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        if data.get('status') == 'error': raise Exception(data.get('message'))
        
        for article in data.get('articles', []):
            link = article.get('url')
            if link:  # We don't check for duplicates here as we'll do it against the master file
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
        texts_to_analyze = [str(row['title']) + ". " + str(row['summary']) for index, row in df.iterrows()]
        
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

def run_backfill(start_date, end_date):
    """ Main function to run the historical data backfill. """
    master_archive_file = "all_historical_news.csv"
    
    # Dummy list of Nifty 50 companies.
    # NOTE: We use a static list here as fetching historical constituents is not feasible.
    nifty50_companies = [
        "Reliance Industries", "TCS", "HDFC Bank", "Infosys", "ICICI Bank", 
        "Hindustan Unilever", "State Bank of India", "Bajaj Finance", "Bharti Airtel", "ITC"
        # Add more company names for better coverage
    ]
    
    # Load existing archive to avoid duplicating entries
    try:
        historical_df = pd.read_csv(master_archive_file)
        seen_urls = set(historical_df['link'])
        logging.info(f"Loaded {len(historical_df)} articles from existing archive.")
    except FileNotFoundError:
        historical_df = pd.DataFrame()
        seen_urls = set()
        logging.info("No existing archive found. Creating a new one.")

    # Loop through each day in the date range
    current_date = start_date
    while current_date <= end_date:
        date_str = current_date.strftime('%Y-%m-%d')
        
        # 1. Fetch Raw News for the day
        
        daily_articles = fetch_historical_newsapi(nifty50_companies, date_str)
        
        if not daily_articles:
            logging.info(f"No articles found for {date_str}. Moving to next day.")
            current_date += timedelta(days=1)
            time.sleep(2) # Sleep even if no articles are found to respect rate limits
            continue
        
        daily_df = pd.DataFrame(daily_articles)
        
        # Filter out articles we already have in our archive
        new_articles_df = daily_df[~daily_df['link'].isin(seen_urls)]
        
        if new_articles_df.empty:
            logging.info(f"No new articles for {date_str}. All found articles are already in the archive.")
        else:
            # 2. Analyze Sentiment for new articles
            analyzed_df = analyze_sentiment(new_articles_df.copy())
            
            # 3. Append to historical DataFrame and update seen_urls
            historical_df = pd.concat([historical_df, analyzed_df], ignore_index=True)
            for link in analyzed_df['link']:
                seen_urls.add(link)
            
            # 4. Save progress to the master file
            historical_df.to_csv(master_archive_file, index=False)
            logging.info(f"SUCCESS: Fetched, analyzed, and saved {len(analyzed_df)} new articles for {date_str}.")

        # Move to the next day and sleep to respect API rate limits
        current_date += timedelta(days=1)
        time.sleep(2) # IMPORTANT: Wait between days to avoid hitting API rate limits

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch historical news for Nifty 50 companies.")
    parser.add_argument("start_date", type=str, help="Start date in YYYY-MM-DD format.")
    parser.add_argument("end_date", type=str, help="End date in YYYY-MM-DD format.")
    args = parser.parse_args()

    start = datetime.strptime(args.start_date, "%Y-%m-%d")
    end = datetime.strptime(args.end_date, "%Y-%m-%d")

    run_backfill(start, end)
    logging.info("Historical backfill process complete.")