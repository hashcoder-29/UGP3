# data_aggregator.py (Updated)
import pandas as pd
from datetime import datetime

def aggregate_options_data(df):
    """
    Calculates key metrics from the aggregated options chain data.
    """
    if df.empty:
        return {}
    
    df['OpenInterest'] = pd.to_numeric(df['OpenInterest'], errors='coerce')
    df['OI_Daily_Change'] = pd.to_numeric(df['OI_Daily_Change'], errors='coerce')
    
    total_call_oi = df[df['OptionType'] == 'Call']['OpenInterest'].sum()
    total_put_oi = df[df['OptionType'] == 'Put']['OpenInterest'].sum()
    pcr = total_put_oi / total_call_oi if total_call_oi > 0 else 0
    
    total_call_oi_change = df[df['OptionType'] == 'Call']['OI_Daily_Change'].sum()
    total_put_oi_change = df[df['OptionType'] == 'Put']['OI_Daily_Change'].sum()
    
    return {
        'total_call_oi': total_call_oi,
        'total_put_oi': total_put_oi,
        'put_call_ratio': pcr,
        'total_call_oi_daily_change': total_call_oi_change, # CHANGED: Added new metric
        'total_put_oi_daily_change': total_put_oi_change    # CHANGED: Added new metric
    }

def aggregate_sentiment_data(df):
    """
    Calculates aggregate sentiment metrics from the news data.
    """
    if df.empty:
        return {}
    
    total_articles = len(df)
    sentiment_counts = df['sentiment_label'].value_counts(normalize=True).to_dict()
    
    pct_positive = sentiment_counts.get('positive', 0.0)
    pct_negative = sentiment_counts.get('negative', 0.0)
    pct_neutral = sentiment_counts.get('neutral', 0.0)
    
    overall_sentiment_score = pct_positive - pct_negative
    
    return {
        'total_articles': total_articles,
        'pct_positive': pct_positive,
        'pct_negative': pct_negative,
        'pct_neutral': pct_neutral,
        'overall_sentiment_score': overall_sentiment_score
    }

if __name__ == "__main__":
    print("--- Starting Data Aggregator ---")
    today_str = datetime.now().strftime('%Y-%m-%d')
    # CHANGED: Use the new aggregated options file
    options_file = f"nifty_options_aggregated_{today_str}.csv" 
    news_file = f"nifty_news_{today_str}.csv"
    summary_file = f"daily_summary_{today_str}.csv"
    
    final_summary = {'date': today_str}
    
    # 1. Process Options Data
    try:
        options_df = pd.read_csv(options_file)
        options_metrics = aggregate_options_data(options_df)
        final_summary.update(options_metrics)
        print(f"Successfully processed options data from {options_file}.")

    except FileNotFoundError:
        print(f"Warning: Options file '{options_file}' not found. Skipping options metrics.")
        
    except Exception as e:
        print(f"Error processing options file: {e}")

    # 2. Process News Sentiment Data
    try:
        news_df = pd.read_csv(news_file)
        sentiment_metrics = aggregate_sentiment_data(news_df)
        final_summary.update(sentiment_metrics)
        print(f"Successfully processed sentiment data from {news_file}.")
    except FileNotFoundError:
        print(f"Warning: News file '{news_file}' not found. Skipping sentiment metrics.")
    except Exception as e:
        print(f"Error processing news file: {e}")
        
    # 3. Create and Save Final Summary
    if len(final_summary) > 1:
        summary_df = pd.DataFrame([final_summary])
        
        # CHANGED: Added new columns to the final summary file
        column_order = [
            'date', 'total_call_oi', 'total_put_oi', 'put_call_ratio',
            'total_call_oi_daily_change', 'total_put_oi_daily_change',
            'total_articles', 'pct_positive', 'pct_negative', 'pct_neutral',
            'overall_sentiment_score'
        ]
        existing_columns = [col for col in column_order if col in summary_df.columns]
        summary_df = summary_df[existing_columns]
        
        summary_df.to_csv(summary_file, index=False)
        print(f"Daily summary successfully saved to {summary_file}")
    else:
        print("No data was processed. Final summary file not created.")
        
    print("--- Data Aggregator Finished ---")