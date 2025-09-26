# sentiment_analyzer.py (Updated with Archiving)
import pandas as pd
from transformers import pipeline
from datetime import datetime
import torch
import os
from db_utils import get_db_engine


def analyze_sentiment(df):
    """
    Analyzes the sentiment of news headlines and summaries using a pre-trained FinBERT model.
    """
    if df.empty:
        print("Input DataFrame is empty. No sentiment analysis to perform.")
        return df

    print("Loading FinBERT model... (This may take a moment on first run)")
    try:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("Device set to use cuda (GPU)")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Device set to use mps (Apple Silicon GPU)")
        else:
            device = torch.device("cpu")
            print("Device set to use cpu")

        sentiment_pipeline = pipeline(
            "sentiment-analysis", 
            model="ProsusAI/finbert",
            device=device
        )
        print("FinBERT model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Cannot perform sentiment analysis.")
        return df
        
    sentiments = []
    print(f"Analyzing sentiment for {len(df)} articles...")
    
    for index, row in df.iterrows():
        text_to_analyze = (str(row['title']) + ". " + str(row['summary']))[:512]
        
        try:
            result = sentiment_pipeline(text_to_analyze)
            sentiment_label = result[0]['label']
            sentiment_score = result[0]['score']
            
            sentiments.append({
                'sentiment_label': sentiment_label,
                'sentiment_score': sentiment_score
            })
        except Exception as e:
            print(f"Could not analyze article {index}: {e}")
            sentiments.append({
                'sentiment_label': 'error',
                'sentiment_score': 0.0
            })
            
    sentiment_df = pd.DataFrame(sentiments)
    df_with_sentiment = pd.concat([df.reset_index(drop=True), sentiment_df], axis=1)
    
    print("Sentiment analysis complete.")
    return df_with_sentiment

if __name__ == "__main__":
    print("--- Starting Sentiment Analyzer ---")
    today_str = datetime.now().strftime('%Y-%m-%d')
    input_filename = f"nifty_news_{today_str}.csv"
    
    analyzed_df = pd.DataFrame() # Initialize an empty dataframe

    try:
        news_df = pd.read_csv(input_filename)
        analyzed_df = analyze_sentiment(news_df)
        
        if 'sentiment_label' not in analyzed_df.columns:
            print("Sentiment analysis failed. Exiting pipeline.")
            exit(1) 
            
        analyzed_df.to_csv(input_filename, index=False)
        print(f"Sentiment analysis results saved back to {input_filename}")

    except FileNotFoundError:
        print(f"Error: Input file '{input_filename}' not found. Please run news_collector.py first.")
        exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        exit(1)

    # --- NEW ARCHIVING LOGIC ---
    if not analyzed_df.empty:
        engine = get_db_engine()
        if engine is not None:
            master_table_name = "historical_news"
            print(f"\n--- Starting News Archiving to Database table: {master_table_name} ---")
            
            try:
                # Check for existing links in the database to prevent duplicates
                existing_links_df = pd.read_sql_query(f'SELECT link FROM {master_table_name}', engine)
                existing_links = set(existing_links_df['link'])
                
                new_articles_df = analyzed_df[~analyzed_df['link'].isin(existing_links)]
                
                if not new_articles_df.empty:
                    # Append only the new articles to the database table
                    new_articles_df.to_sql(master_table_name, engine, if_exists='append', index=False)
                    print(f"✅ Appended {len(new_articles_df)} new articles to the database.")
                else:
                    print("No new articles to add to the database.")
            except Exception as e:
                print(f"Error archiving news to database: {e}")
    # ---------------------------
    
    print("\n--- Sentiment Analyzer Finished ---")