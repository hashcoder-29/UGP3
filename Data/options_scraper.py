# options_scraper.py
import requests
import pandas as pd
from datetime import datetime
import time

def get_nse_cookies():
    """
    Fetches the necessary cookies from the NSE India homepage to make subsequent API calls.
    The NSE website requires initial cookies to be set before accepting API requests.
    """
    try:
        # URL of the main page to initialize session
        url = 'https://www.nseindia.com/option-chain'
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
        }
        # A session object is used to persist cookies across requests
        session = requests.Session()
        session.get(url, headers=headers)
        return session
    except requests.exceptions.RequestException as e:
        print(f"Error fetching initial NSE cookies: {e}")
        return None

def fetch_options_data(session):
    """
    Fetches the Nifty 50 options chain data using the provided session with cookies.
    """
    if not session:
        print("Session not available. Cannot fetch options data.")
        return None
        
    # The actual API endpoint for the Nifty options chain
    api_url = "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
    }
    
    try:
        print("Fetching Nifty 50 options chain data...")
        response = session.get(api_url, headers=headers)
        response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)
        data = response.json()
        print("Successfully fetched options data.")
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching options data from NSE API: {e}")
        return None

def process_options_data(data):
    """
    Processes the raw JSON data into a structured Pandas DataFrame.
    """
    if not data or 'records' not in data or 'data' not in data['records']:
        print("No valid data to process.")
        return pd.DataFrame()

    records = data['records']['data']
    processed_data = []

    for record in records:
        # Each record contains both CE and PE data if available
        if 'CE' in record:
            ce_data = record['CE']
            processed_data.append({
                'ExpiryDate': ce_data.get('expiryDate'),
                'OptionType': 'Call',
                'StrikePrice': ce_data.get('strikePrice'),
                'OpenInterest': ce_data.get('openInterest'),
                'ChangeInOI': ce_data.get('changeinOpenInterest'),
                'LTP': ce_data.get('lastPrice'),
                'IV': ce_data.get('impliedVolatility'),
            })
        
        if 'PE' in record:
            pe_data = record['PE']
            processed_data.append({
                'ExpiryDate': pe_data.get('expiryDate'),
                'OptionType': 'Put',
                'StrikePrice': pe_data.get('strikePrice'),
                'OpenInterest': pe_data.get('openInterest'),
                'ChangeInOI': pe_data.get('changeinOpenInterest'),
                'LTP': pe_data.get('lastPrice'),
                'IV': pe_data.get('impliedVolatility'),
            })

    df = pd.DataFrame(processed_data)
    return df

if __name__ == "__main__":
    print("--- Starting Options Data Scraper ---")
    
    # 1. Get session cookies
    nse_session = get_nse_cookies()
    
    # Add a small delay to mimic human behavior
    time.sleep(2)

    # 2. Fetch data
    options_json = fetch_options_data(nse_session)
    
    if options_json:
        # 3. Process data
        options_df = process_options_data(options_json)
        
        if not options_df.empty:
            # 4. Save data to CSV
            today_str = datetime.now().strftime('%Y-%m-%d')
            filename = f"nifty_options_{today_str}.csv"
            options_df.to_csv(filename, index=False)
            print(f"Options data successfully saved to {filename}")
        else:
            print("Failed to process options data. Output file not created.")
    else:
        print("Failed to fetch options data.")

    print("--- Options Data Scraper Finished ---")