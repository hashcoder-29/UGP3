# options_aggregator.py
import pandas as pd
from datetime import datetime, timedelta

def aggregate_daily_options_data():
    """
    Loads today's and yesterday's options data to calculate daily changes.
    """
    today = datetime.now()
    yesterday = today - timedelta(days=1)
    
    today_str = today.strftime('%Y-%m-%d')
    yesterday_str = yesterday.strftime('%Y-%m-%d')

    today_filename = f"nifty_options_{today_str}.csv"
    yesterday_filename = f"nifty_options_{yesterday_str}.csv"
    output_filename = f"nifty_options_aggregated_{today_str}.csv"

    print(f"Reading today's data from: {today_filename}")
    try:
        today_df = pd.read_csv(today_filename)
    except FileNotFoundError:
        print(f"Error: Today's data file not found: {today_filename}")
        print("Please run options_scraper.py first. Exiting.")
        return # Use return instead of exit(1) to be friendlier in complex pipelines

    try:
        print(f"Reading previous day's data from: {yesterday_filename}")
        yesterday_df = pd.read_csv(yesterday_filename)
        
        # Prepare yesterday's data for merging
        # We only need the key columns and the values we want to compare
        yesterday_subset = yesterday_df[[
            'StrikePrice', 'ExpiryDate', 'OptionType', 'OpenInterest', 'IV'
        ]].rename(columns={
            'OpenInterest': 'OpenInterest_prev',
            'IV': 'IV_prev'
        })

        # Merge today's data with yesterday's data
        # A 'left' merge keeps all of today's options, even if they were not present yesterday
        merged_df = pd.merge(
            today_df,
            yesterday_subset,
            on=['StrikePrice', 'ExpiryDate', 'OptionType'],
            how='left'
        )
        
        # Fill missing previous-day values with 0, assuming they are new contracts
        merged_df['OpenInterest_prev'] = merged_df['OpenInterest_prev'].fillna(0)
        merged_df['IV_prev'] = merged_df['IV_prev'].fillna(0)

        # Calculate the daily change
        merged_df['OI_Daily_Change'] = merged_df['OpenInterest'] - merged_df['OpenInterest_prev']
        merged_df['IV_Daily_Change'] = merged_df['IV'] - merged_df['IV_prev']
        
        print("Successfully calculated daily changes.")
        merged_df.to_csv(output_filename, index=False)
        print(f"Aggregated options data saved to {output_filename}")

    except FileNotFoundError:
        print(f"Warning: Previous day's data file not found: {yesterday_filename}")
        print("This may be the first run. Saving today's data with 'change' columns set to 0.")
        
        # If yesterday's file doesn't exist, create the change columns with zero values
        today_df['OI_Daily_Change'] = today_df['ChangeInOI'] # Use today's change as the daily change
        today_df['IV_Daily_Change'] = 0.0
        
        today_df.to_csv(output_filename, index=False)
        print(f"Aggregated options data saved to {output_filename}")


if __name__ == "__main__":
    print("--- Starting Daily Options Aggregator ---")
    aggregate_daily_options_data()
    print("--- Daily Options Aggregator Finished ---")