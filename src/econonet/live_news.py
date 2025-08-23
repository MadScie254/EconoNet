"""
EconoNet Live News & Insights Module

Fetches and normalizes fintech/finance/crypto news from multiple sources.
Returns standardized DataFrames with sentiment analysis and fallback support.
"""

import pandas as pd
import numpy as np
import requests
import json
import time
import logging
import re
import warnings
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from urllib3.util.retry import Retry

# Optional imports with fallbacks
try:
    import feedparser
    FEEDPARSER_AVAILABLE = True
except ImportError:
    FEEDPARSER_AVAILABLE = False

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

try:
    from requests.adapters import HTTPAdapter
except ImportError:
    HTTPAdapter = None

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class NewsConfig:
    """Configuration for news API endpoints"""
    gnews_base_url: str = "https://gnews.io/api/v4"
    yahoo_finance_rss: str = "https://finance.yahoo.com/rss"
    cryptopanic_base_url: str = "https://cryptopanic.com/api/v1"
    timeout_seconds: int = 30
    max_retries: int = 3
    cache_ttl_minutes: int = 15
    max_articles_per_source: int = 50

# Global configuration
news_config = NewsConfig()

def get_session() -> requests.Session:
    """Create a requests session with retry strategy"""
    session = requests.Session()
    
    if HTTPAdapter is not None:
        try:
            retry_strategy = Retry(
                total=news_config.max_retries,
                status_forcelist=[429, 500, 502, 503, 504],
                method_whitelist=["HEAD", "GET", "OPTIONS"],
                backoff_factor=1
            )
            adapter = HTTPAdapter(max_retries=retry_strategy)
            session.mount("http://", adapter)
            session.mount("https://", adapter)
        except Exception as e:
            logger.warning(f"Could not configure retry adapter: {e}")
    
    return session
    return session

def analyze_sentiment(text: str) -> tuple:
    """
    Analyze sentiment of text using TextBlob (with fallback)
    Returns (sentiment_score, sentiment_label, emoji)
    """
    if not text or pd.isna(text):
        return 0.0, 'neutral', '⚪'
    
    if not TEXTBLOB_AVAILABLE:
        # Simple rule-based fallback sentiment analysis
        text_lower = str(text).lower()
        positive_words = ['growth', 'up', 'rise', 'gain', 'positive', 'strong', 'bullish', 'high', 'good', 'success', 'profit']
        negative_words = ['down', 'fall', 'drop', 'loss', 'negative', 'weak', 'bearish', 'low', 'bad', 'crash', 'decline']
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            return 0.5, 'bullish', '🟢'
        elif negative_count > positive_count:
            return -0.5, 'bearish', '🔴'
        else:
            return 0.0, 'neutral', '⚪'
    
    try:
        blob = TextBlob(str(text))
        polarity = blob.sentiment.polarity
        
        if polarity > 0.1:
            return polarity, 'bullish', '🟢'
        elif polarity < -0.1:
            return polarity, 'bearish', '🔴'
        else:
            return polarity, 'neutral', '⚪'
    except Exception as e:
        logger.warning(f"Sentiment analysis failed: {e}")
        return 0.0, 'neutral', '⚪'

def generate_fallback_news() -> pd.DataFrame:
    """Generate sample news data when APIs are unavailable"""
    sample_news = [
        {
            'date': datetime.now() - timedelta(hours=1),
            'title': 'African Fintech Sector Shows Strong Growth in Q3 2025',
            'source': 'EconoNet Fallback',
            'url': 'https://example.com/african-fintech-growth',
            'snippet': 'Kenya and Nigeria lead the charge in mobile payments and digital banking adoption across the continent.',
            'category': 'fintech',
            'region': 'africa'
        },
        {
            'date': datetime.now() - timedelta(hours=3),
            'title': 'Central Bank Digital Currency Trials Expand in East Africa',
            'source': 'EconoNet Fallback',
            'url': 'https://example.com/cbdc-trials',
            'snippet': 'Tanzania and Uganda join Kenya in testing digital currency frameworks for cross-border payments.',
            'category': 'banking',
            'region': 'africa'
        },
        {
            'date': datetime.now() - timedelta(hours=6),
            'title': 'Bitcoin Volatility Stabilizes as Institutional Adoption Grows',
            'source': 'EconoNet Fallback',
            'url': 'https://example.com/bitcoin-stabilization',
            'snippet': 'Major African pension funds begin allocation strategies for cryptocurrency investments.',
            'category': 'crypto',
            'region': 'global'
        },
        {
            'date': datetime.now() - timedelta(hours=12),
            'title': 'New Fintech Regulations Promote Innovation in Ghana',
            'source': 'EconoNet Fallback',
            'url': 'https://example.com/ghana-regulations',
            'snippet': 'Regulatory sandbox allows startups to test financial services with reduced compliance burden.',
            'category': 'regulation',
            'region': 'africa'
        },
        {
            'date': datetime.now() - timedelta(days=1),
            'title': 'Mobile Money Transactions Reach Record High in Kenya',
            'source': 'EconoNet Fallback',
            'url': 'https://example.com/mobile-money-kenya',
            'snippet': 'M-Pesa processes over $2 billion in monthly transactions, showcasing digital payment adoption.',
            'category': 'fintech',
            'region': 'africa'
        }
    ]
    
    df = pd.DataFrame(sample_news)
    
    # Add sentiment analysis
    sentiment_data = []
    for _, row in df.iterrows():
        score, label, emoji = analyze_sentiment(row['snippet'])
        sentiment_data.append({
            'sentiment_score': score,
            'sentiment_label': label,
            'sentiment_emoji': emoji
        })
    
    sentiment_df = pd.DataFrame(sentiment_data)
    result = pd.concat([df, sentiment_df], axis=1)
    
    # Add metadata
    result['fallback'] = True
    result['last_refresh'] = datetime.now().isoformat()
    
    return result

def get_gnews(query: str = "fintech", lang: str = "en", country: str = "us") -> pd.DataFrame:
    """
    Fetch news from GNews API (free tier - no token required)
    
    Args:
        query: Search query (default: "fintech")
        lang: Language code (default: "en")
        country: Country code (default: "us")
    
    Returns:
        DataFrame with news articles
    """
    try:
        # Use free GNews search endpoint (no API key required for basic search)
        url = f"https://gnews.io/api/v4/search"
        params = {
            'q': query,
            'lang': lang,
            'country': country,
            'max': news_config.max_articles_per_source,
            'sortby': 'publishedAt'
        }
        
        session = get_session()
        response = session.get(url, params=params, timeout=news_config.timeout_seconds)
        
        # If GNews API requires key, fallback to RSS approach
        if response.status_code == 401:
            logger.warning("GNews API requires authentication, using fallback RSS")
            return get_rss_feed("https://news.google.com/rss/search?q=fintech&hl=en&gl=us")
        
        response.raise_for_status()
        data = response.json()
        
        articles = []
        for article in data.get('articles', []):
            articles.append({
                'date': pd.to_datetime(article.get('publishedAt', datetime.now())),
                'title': article.get('title', ''),
                'source': article.get('source', {}).get('name', 'GNews'),
                'url': article.get('url', ''),
                'snippet': article.get('description', ''),
                'category': 'fintech',
                'region': 'global'
            })
        
        if not articles:
            logger.warning("No articles from GNews, using fallback")
            return generate_fallback_news()
        
        df = pd.DataFrame(articles)
        
        # Add sentiment analysis
        sentiment_data = []
        for _, row in df.iterrows():
            score, label, emoji = analyze_sentiment(row['snippet'])
            sentiment_data.append({
                'sentiment_score': score,
                'sentiment_label': label,
                'sentiment_emoji': emoji
            })
        
        sentiment_df = pd.DataFrame(sentiment_data)
        result = pd.concat([df, sentiment_df], axis=1)
        
        # Add metadata
        result['fallback'] = False
        result['last_refresh'] = datetime.now().isoformat()
        
        return result
        
    except Exception as e:
        logger.error(f"GNews API error: {e}")
        return generate_fallback_news()

def get_yahoo_finance_feed(ticker: str = "BTC-USD") -> pd.DataFrame:
    """
    Fetch news from Yahoo Finance RSS feed
    
    Args:
        ticker: Financial instrument ticker (default: "BTC-USD")
    
    Returns:
        DataFrame with financial news
    """
    try:
        # Yahoo Finance RSS feed for specific ticker
        url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
        
        session = get_session()
        response = session.get(url, timeout=news_config.timeout_seconds)
        response.raise_for_status()
        
        if not FEEDPARSER_AVAILABLE:
            logger.warning("feedparser not available, using fallback RSS parsing")
            return generate_fallback_news()
        
        feed = feedparser.parse(response.content)
        
        articles = []
        for entry in feed.entries[:news_config.max_articles_per_source]:
            # Parse date
            pub_date = datetime.now()
            if hasattr(entry, 'published_parsed') and entry.published_parsed:
                pub_date = datetime(*entry.published_parsed[:6])
            
            articles.append({
                'date': pub_date,
                'title': entry.get('title', ''),
                'source': f"Yahoo Finance ({ticker})",
                'url': entry.get('link', ''),
                'snippet': entry.get('summary', ''),
                'category': 'finance',
                'region': 'global'
            })
        
        if not articles:
            logger.warning("No articles from Yahoo Finance, using fallback")
            return generate_fallback_news()
        
        df = pd.DataFrame(articles)
        
        # Add sentiment analysis
        sentiment_data = []
        for _, row in df.iterrows():
            score, label, emoji = analyze_sentiment(row['snippet'])
            sentiment_data.append({
                'sentiment_score': score,
                'sentiment_label': label,
                'sentiment_emoji': emoji
            })
        
        sentiment_df = pd.DataFrame(sentiment_data)
        result = pd.concat([df, sentiment_df], axis=1)
        
        # Add metadata
        result['fallback'] = False
        result['last_refresh'] = datetime.now().isoformat()
        
        return result
        
    except Exception as e:
        logger.error(f"Yahoo Finance RSS error: {e}")
        return generate_fallback_news()

def get_cryptopanic_feed(filter_type: str = "news") -> pd.DataFrame:
    """
    Fetch crypto news from CryptoPanic RSS (free tier)
    
    Args:
        filter_type: Type of content ("news", "media", "all")
    
    Returns:
        DataFrame with crypto news
    """
    try:
        # CryptoPanic RSS feed (free, no API key required)
        url = f"https://cryptopanic.com/api/v1/posts/?auth_token=free&filter={filter_type}&format=rss"
        
        session = get_session()
        response = session.get(url, timeout=news_config.timeout_seconds)
        response.raise_for_status()
        
        if not FEEDPARSER_AVAILABLE:
            logger.warning("feedparser not available, using fallback RSS parsing")
            return generate_fallback_news()
        
        feed = feedparser.parse(response.content)
        
        articles = []
        for entry in feed.entries[:news_config.max_articles_per_source]:
            # Parse date
            pub_date = datetime.now()
            if hasattr(entry, 'published_parsed') and entry.published_parsed:
                pub_date = datetime(*entry.published_parsed[:6])
            
            articles.append({
                'date': pub_date,
                'title': entry.get('title', ''),
                'source': 'CryptoPanic',
                'url': entry.get('link', ''),
                'snippet': entry.get('summary', ''),
                'category': 'crypto',
                'region': 'global'
            })
        
        if not articles:
            logger.warning("No articles from CryptoPanic, using fallback")
            return generate_fallback_news()
        
        df = pd.DataFrame(articles)
        
        # Add sentiment analysis
        sentiment_data = []
        for _, row in df.iterrows():
            score, label, emoji = analyze_sentiment(row['snippet'])
            sentiment_data.append({
                'sentiment_score': score,
                'sentiment_label': label,
                'sentiment_emoji': emoji
            })
        
        sentiment_df = pd.DataFrame(sentiment_data)
        result = pd.concat([df, sentiment_df], axis=1)
        
        # Add metadata
        result['fallback'] = False
        result['last_refresh'] = datetime.now().isoformat()
        
        return result
        
    except Exception as e:
        logger.error(f"CryptoPanic RSS error: {e}")
        return generate_fallback_news()

def get_rss_feed(url: str, source_name: str = None) -> pd.DataFrame:
    """
    Parse generic RSS feed and return standardized DataFrame
    
    Args:
        url: RSS feed URL
        source_name: Name to use for source (default: extracted from URL)
    
    Returns:
        DataFrame with parsed RSS articles
    """
    try:
        if source_name is None:
            # Extract domain name from URL
            import urllib.parse
            parsed_url = urllib.parse.urlparse(url)
            source_name = parsed_url.netloc.replace('www.', '')
        
        session = get_session()
        response = session.get(url, timeout=news_config.timeout_seconds)
        response.raise_for_status()
        
        if not FEEDPARSER_AVAILABLE:
            logger.warning("feedparser not available, using fallback RSS parsing")
            return generate_fallback_news()
        
        feed = feedparser.parse(response.content)
        
        articles = []
        for entry in feed.entries[:news_config.max_articles_per_source]:
            # Parse date
            pub_date = datetime.now()
            if hasattr(entry, 'published_parsed') and entry.published_parsed:
                pub_date = datetime(*entry.published_parsed[:6])
            
            # Determine category from title/content
            title_lower = entry.get('title', '').lower()
            category = 'fintech'  # default
            if any(word in title_lower for word in ['bitcoin', 'crypto', 'blockchain', 'ethereum']):
                category = 'crypto'
            elif any(word in title_lower for word in ['bank', 'central bank', 'monetary']):
                category = 'banking'
            elif any(word in title_lower for word in ['regulation', 'policy', 'law', 'compliance']):
                category = 'regulation'
            
            articles.append({
                'date': pub_date,
                'title': entry.get('title', ''),
                'source': source_name,
                'url': entry.get('link', ''),
                'snippet': entry.get('summary', ''),
                'category': category,
                'region': 'global'
            })
        
        if not articles:
            logger.warning(f"No articles from RSS feed {url}, using fallback")
            return generate_fallback_news()
        
        df = pd.DataFrame(articles)
        
        # Add sentiment analysis
        sentiment_data = []
        for _, row in df.iterrows():
            score, label, emoji = analyze_sentiment(row['snippet'])
            sentiment_data.append({
                'sentiment_score': score,
                'sentiment_label': label,
                'sentiment_emoji': emoji
            })
        
        sentiment_df = pd.DataFrame(sentiment_data)
        result = pd.concat([df, sentiment_df], axis=1)
        
        # Add metadata
        result['fallback'] = False
        result['last_refresh'] = datetime.now().isoformat()
        
        return result
        
    except Exception as e:
        logger.error(f"RSS feed error for {url}: {e}")
        return generate_fallback_news()

def get_aggregated_news(query: str = "fintech", categories: List[str] = None) -> pd.DataFrame:
    """
    Aggregate news from multiple sources
    
    Args:
        query: Search query
        categories: List of categories to filter ('fintech', 'crypto', 'banking', 'regulation')
    
    Returns:
        Combined DataFrame from all sources
    """
    if categories is None:
        categories = ['fintech', 'crypto', 'banking', 'regulation']
    
    all_news = []
    
    # Fetch from multiple sources
    sources = [
        ("GNews", lambda: get_gnews(query)),
        ("Yahoo Finance", lambda: get_yahoo_finance_feed("BTC-USD")),
        ("CryptoPanic", lambda: get_cryptopanic_feed("news")),
        ("Reuters RSS", lambda: get_rss_feed("https://www.reuters.com/business/finance/rss", "Reuters Finance")),
        ("BBC Business RSS", lambda: get_rss_feed("http://feeds.bbci.co.uk/news/business/rss.xml", "BBC Business"))
    ]
    
    for source_name, fetch_func in sources:
        try:
            logger.info(f"Fetching news from {source_name}")
            df = fetch_func()
            if not df.empty:
                all_news.append(df)
                time.sleep(1)  # Rate limiting
        except Exception as e:
            logger.warning(f"Failed to fetch from {source_name}: {e}")
            continue
    
    if not all_news:
        logger.warning("All news sources failed, using fallback")
        return generate_fallback_news()
    
    # Combine all sources
    combined_df = pd.concat(all_news, ignore_index=True)
    
    # Filter by categories if specified
    if categories:
        combined_df = combined_df[combined_df['category'].isin(categories)]
    
    # Remove duplicates based on title similarity
    combined_df = combined_df.drop_duplicates(subset=['title'], keep='first')
    
    # Sort by date (newest first)
    combined_df = combined_df.sort_values('date', ascending=False)
    
    # Limit total articles
    combined_df = combined_df.head(100)
    
    return combined_df.reset_index(drop=True)

def search_news(query: str, region: str = "global", sentiment: str = "all") -> pd.DataFrame:
    """
    Search news with filters
    
    Args:
        query: Search query
        region: Region filter ("global", "africa", "kenya")
        sentiment: Sentiment filter ("all", "bullish", "bearish", "neutral")
    
    Returns:
        Filtered DataFrame
    """
    # Get aggregated news
    df = get_aggregated_news(query)
    
    # Apply region filter
    if region != "global":
        df = df[df['region'] == region]
    
    # Apply sentiment filter
    if sentiment != "all":
        df = df[df['sentiment_label'] == sentiment]
    
    return df

def get_news_summary_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate summary statistics for news data
    
    Args:
        df: News DataFrame
    
    Returns:
        Dictionary with summary statistics
    """
    if df.empty:
        return {
            'total_articles': 0,
            'sources': 0,
            'sentiment_distribution': {},
            'category_distribution': {},
            'latest_article': None,
            'avg_sentiment': 0.0
        }
    
    return {
        'total_articles': len(df),
        'sources': df['source'].nunique(),
        'sentiment_distribution': df['sentiment_label'].value_counts().to_dict(),
        'category_distribution': df['category'].value_counts().to_dict(),
        'latest_article': df.iloc[0]['date'].isoformat() if len(df) > 0 else None,
        'avg_sentiment': df['sentiment_score'].mean(),
        'fallback_ratio': df['fallback'].mean() if 'fallback' in df.columns else 0.0
    }

# Main aggregation function for easy import
def get_fintech_news(query: str = "fintech", region: str = "global", 
                     categories: List[str] = None, limit: int = 50) -> pd.DataFrame:
    """
    Main function to get fintech news with all features
    
    Args:
        query: Search query
        region: Region filter
        categories: Category filters
        limit: Maximum number of articles
    
    Returns:
        Processed news DataFrame
    """
    try:
        df = get_aggregated_news(query, categories)
        
        # Apply region filter
        if region != "global":
            df = df[df['region'] == region]
        
        # Limit results
        df = df.head(limit)
        
        return df
        
    except Exception as e:
        logger.error(f"Error getting fintech news: {e}")
        return generate_fallback_news()

if __name__ == "__main__":
    # Test the module
    print("Testing EconoNet Live News Module...")
    
    # Test basic functionality
    news_df = get_fintech_news("fintech africa", region="global", limit=10)
    print(f"Retrieved {len(news_df)} articles")
    print(f"Columns: {list(news_df.columns)}")
    
    if not news_df.empty:
        print(f"\nSample article:")
        print(f"Title: {news_df.iloc[0]['title']}")
        print(f"Source: {news_df.iloc[0]['source']}")
        print(f"Sentiment: {news_df.iloc[0]['sentiment_emoji']} {news_df.iloc[0]['sentiment_label']}")
    
    # Test summary stats
    stats = get_news_summary_stats(news_df)
    print(f"\nSummary stats: {stats}")
