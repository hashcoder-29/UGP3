# news_collector.py (v6 - India Focus Edition)
import pandas as pd
import requests
from bs4 import BeautifulSoup
import feedparser
from datetime import datetime, timedelta
import time
import random
import logging
# ##############################################################################
# ## PASTE YOUR FIVE FREE API KEYS HERE                                       ##
# ##############################################################################
NEWS_API_KEY = "6532d685925c4cb0bc86026474157de2"         # From https://newsapi.org/
# ##############################################################################

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


def fetch_news_from_rss(keywords, seen_urls):
    """
    Fetches news from an expanded list of RSS feeds.
    """
    rss_feeds = {
        'Economic Times': 'https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms',
        'Moneycontrol': 'https://www.moneycontrol.com/rss/business.xml',
        'Livemint': 'https://www.livemint.com/rss/markets',
        # Remove Reuters or scrape it using BeautifulSoup instead
    }

    
    articles = []
    print("\nFetching news from RSS feeds...")
    for source, url in rss_feeds.items():
        try:
            feed = feedparser.parse(url, request_headers={'User-Agent': 'Mozilla/5.0'})
            print(f"[DEBUG] RSS Feed: {source}, Entries found: {len(feed.entries)}")
            if len(feed.entries) == 0:
                print(f"[DEBUG] RSS Feed content (first 200 chars): {feed}")
            for entry in feed.entries:
                title = entry.get('title', '')
                summary = entry.get('summary', '')
                link = entry.get('link', '')
                pub_date = entry.get('published', datetime.now().isoformat())
                
                if not link or link in seen_urls:
                    continue

                content_to_check = f"{title} {summary}".lower()
                if any(keyword.lower() in content_to_check for keyword in keywords):
                    articles.append({
                        'source': source,
                        'title': title,
                        'summary': summary,
                        'link': link,
                        'published_date': pub_date
                    })
                    seen_urls.add(link)
            print(f"  > Found {len(feed.entries)} articles from {source}.")
            time.sleep(1)
        except Exception as e:
            print(f"  > Could not fetch from {source}: {e}")
    return articles

# --- This function remains the same ---
def scrape_moneycontrol_news(keywords, seen_urls):
    """
    Scrapes the Moneycontrol market news page to find relevant articles.
    """
    articles = []
    print("\nScraping news from Moneycontrol website...")
    try:
        url = "https://www.moneycontrol.com/news/business/markets/"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'}
        response = requests.get(url, headers=headers)
        print(f"[DEBUG] Moneycontrol Status: {response.status_code}")
        print(f"[DEBUG] Moneycontrol HTML Sample: {response.text[:300]}")
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        news_list = soup.select('ul.clearfix li')
        
        for item in news_list:
            title_element = item.select_one('h2 > a')
            summary_element = item.select_one('p')
            
            if title_element and title_element.has_attr('title') and title_element.has_attr('href'):
                title = title_element['title']
                link = title_element['href']
                summary = summary_element.get_text(strip=True) if summary_element else ""
                
                if not link or link in seen_urls:
                    continue
                
                content_to_check = f"{title} {summary}".lower()
                if any(keyword.lower() in content_to_check for keyword in keywords):
                    articles.append({
                        'source': 'Moneycontrol Web',
                        'title': title,
                        'summary': summary,
                        'link': link,
                        'published_date': datetime.now().isoformat()
                    })
                    seen_urls.add(link)
        print(f"  > Scraped and filtered {len(articles)} relevant articles from Moneycontrol.")
    except Exception as e:
        print(f"  > Could not scrape Moneycontrol: {e}")
    return articles

def fetch_news_from_newsapi(company_names, seen_urls):
    if not NEWS_API_KEY: return []
    articles = []
    print("\nFetching news from NewsAPI...")
    for name in company_names[:5]: # Query a subset to respect API limits
        try:
            query = f'"{name}"'
            url = f"https://newsapi.org/v2/everything?q={query}&language=en&sortBy=publishedAt&apiKey={NEWS_API_KEY}"
            response = requests.get(url)
            data = response.json()
            if data.get('status') == 'error': raise Exception(data.get('message'))
            for article in data.get('articles', []):
                link = article.get('url')
                if link and link not in seen_urls:
                    articles.append({'source': f"NewsAPI - {article.get('source', {}).get('name')}", 'title': article.get('title'), 'summary': article.get('description'), 'link': link, 'published_date': article.get('publishedAt')})
                    seen_urls.add(link)
            time.sleep(1)
        except Exception as e: print(f"  > Error (NewsAPI, {name}): {e}")
    print(f"  > Found {len(articles)} new articles from NewsAPI.")
    return articles
# --- New Web Scraping Function for The Economic Times ---
def scrape_economic_times_news(keywords, seen_urls):
    """
    Scrapes The Economic Times market news page for relevant articles.
    """
    articles = []
    print("\nScraping news from The Economic Times website...")
    try:
        base_url = "https://economictimes.indiatimes.com"
        url = f"{base_url}/markets/stocks/news"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # This selector targets the main news list container on ET Markets
        news_list = soup.select('div.newsList a')
        
        for item in news_list:
            title = item.get_text(strip=True)
            link = item.get('href')
            
            if title and link:
                # ET links are often relative, so we construct the full URL
                if not link.startswith('http'):
                    link = base_url + link
                
                if link in seen_urls:
                    continue
                
                content_to_check = title.lower() # ET list view doesn't have summaries, so we check title
                if any(keyword.lower() in content_to_check for keyword in keywords):
                    articles.append({
                        'source': 'Economic Times Web',
                        'title': title,
                        'summary': '', # No summary available in this view
                        'link': link,
                        'published_date': datetime.now().isoformat()
                    })
                    seen_urls.add(link)
        print(f"  > Scraped and filtered {len(articles)} relevant articles from The Economic Times.")
    except Exception as e:
        print(f"  > Could not scrape The Economic Times: {e}")
    return articles

def scrape_livemint_news(keywords, seen_urls):
    """ Scrapes the Livemint market news page for relevant articles. """
    articles = []
    print("\nScraping news from Livemint website...")
    try:
        url = "https://www.livemint.com/market/stock-market-news"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # This selector targets headline links within the main content area
        news_list = soup.select('div.listtostory h2 a')
        
        for item in news_list:
            title = item.get_text(strip=True)
            link = item.get('href')
            if not link.startswith('http'):
                link = "https://www.livemint.com" + link

            if link and link not in seen_urls:
                content_to_check = title.lower()
                if any(keyword.lower() in content_to_check for keyword in keywords):
                    articles.append({'source': 'Livemint Web', 'title': title, 'summary': '', 'link': link, 'published_date': datetime.now().isoformat()})
                    seen_urls.add(link)
        print(f"  > Scraped and filtered {len(articles)} relevant articles from Livemint.")
    except Exception as e:
        print(f"  > Could not scrape Livemint: {e}")
    return articles

if __name__ == "__main__":
    print("--- Starting News Collector (v8 - Refactored Edition) ---")
    nse_session = get_nse_session()
    
    if nse_session:
        constituents = get_nifty50_constituents(nse_session) # Pass the session
        print(constituents)
    else:
        logging.error("Could not create NSE session. Using fallback for constituents.")
        constituents = {"RELIANCE": "Reliance Industries Ltd.", "TCS": "Tata Consultancy Services Ltd."}

    company_symbols = list(constituents.keys())
    company_names = list(constituents.values())
        
    av_symbols_sample = random.sample(company_symbols, min(len(company_symbols), 10))
    
    seen_urls = set()
    all_articles = []
    base_keywords = ["Nifty 50", "NSE", "Sensex", "Indian market", "RBI"]
    all_keywords = base_keywords + company_names
    



    all_articles.extend(fetch_news_from_newsapi(company_names, seen_urls)) # Now India-focused
    all_articles.extend(scrape_livemint_news(all_keywords, seen_urls))
    rss_articles = fetch_news_from_rss(all_keywords, seen_urls)
    all_articles.extend(rss_articles)
    
    # 2. Fetch via web scraping from Moneycontrol
    mc_articles = scrape_moneycontrol_news(all_keywords, seen_urls)
    all_articles.extend(mc_articles)
    
    # ## --- NEW --- ##
    # 3. Fetch via web scraping from Economic Times
    et_articles = scrape_economic_times_news(all_keywords, seen_urls)
    all_articles.extend(et_articles)
    
    # 4. Final processing and saving
    if all_articles:
        print(f"\nCollected a total of {len(all_articles)} unique relevant news articles from all sources.")
        news_df = pd.DataFrame(all_articles)
        today_str = datetime.now().strftime('%Y-%m-%d')
        filename = f"nifty_news_{today_str}.csv"
        news_df.to_csv(filename, index=False)
        print(f"News data successfully saved to {filename}")
    else:
        print("\n No relevant news articles were collected. Output file not created.")

    print("--- News Collector Finished ---")