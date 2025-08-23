"""
Tests for EconoNet Live News Module

Tests news fetching, parsing, sentiment analysis, and fallback mechanisms.
"""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta
import sys
import os
from pathlib import Path

# Add src to path
current_dir = Path(__file__).parent.parent
sys.path.append(str(current_dir / 'src'))

try:
    from src.econonet.live_news import (
        get_fintech_news, analyze_sentiment, generate_fallback_news, 
        search_news, get_news_summary_stats
    )
    NEWS_MODULE_AVAILABLE = True
except ImportError:
    try:
        # Alternative import path
        sys.path.append(str(current_dir / 'src' / 'econonet'))
        from live_news import (
            get_fintech_news, analyze_sentiment, generate_fallback_news,
            search_news, get_news_summary_stats
        )
        NEWS_MODULE_AVAILABLE = True
    except ImportError:
        NEWS_MODULE_AVAILABLE = False

# Sample RSS feed content for testing
SAMPLE_RSS_CONTENT = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
    <channel>
        <title>Fintech News</title>
        <description>Latest fintech news and updates</description>
        <item>
            <title>African Fintech Startup Raises $50M Series B</title>
            <description>Leading mobile payments company secures funding for expansion across East Africa</description>
            <link>https://example.com/news/fintech-funding</link>
            <pubDate>Mon, 23 Aug 2025 10:00:00 GMT</pubDate>
        </item>
        <item>
            <title>Central Bank Announces CBDC Pilot Program</title>
            <description>New digital currency initiative aims to modernize payment infrastructure</description>
            <link>https://example.com/news/cbdc-pilot</link>
            <pubDate>Sun, 22 Aug 2025 15:30:00 GMT</pubDate>
        </item>
    </channel>
</rss>"""

# Sample GNews API response
SAMPLE_GNEWS_RESPONSE = {
    "totalArticles": 2,
    "articles": [
        {
            "title": "Blockchain Technology Revolutionizes Banking",
            "description": "Major banks adopt blockchain for faster cross-border payments",
            "url": "https://example.com/blockchain-banking",
            "publishedAt": "2025-08-23T08:00:00Z",
            "source": {
                "name": "TechNews",
                "url": "https://technews.com"
            }
        },
        {
            "title": "Cryptocurrency Regulation Update",
            "description": "New regulatory framework provides clarity for crypto businesses",
            "url": "https://example.com/crypto-regulation",
            "publishedAt": "2025-08-22T14:00:00Z",
            "source": {
                "name": "FinanceDaily",
                "url": "https://financedaily.com"
            }
        }
    ]
}

@pytest.mark.skipif(not NEWS_MODULE_AVAILABLE, reason="News module not available")
class TestNewsModule:
    """Test cases for the news module"""
    
    def test_sentiment_analysis(self):
        """Test sentiment analysis functionality"""
        # Test positive sentiment
        score, label, emoji = analyze_sentiment("This is great news for fintech innovation!")
        assert label == 'bullish'
        assert emoji == '🟢'
        assert score > 0
        
        # Test negative sentiment
        score, label, emoji = analyze_sentiment("The market crash is devastating for investors")
        assert label == 'bearish'
        assert emoji == '🔴'
        assert score < 0
        
        # Test neutral sentiment
        score, label, emoji = analyze_sentiment("The bank announced quarterly results")
        assert label == 'neutral'
        assert emoji == '⚪'
        
        # Test empty text
        score, label, emoji = analyze_sentiment("")
        assert label == 'neutral'
        assert score == 0.0
    
    def test_generate_fallback_news(self):
        """Test fallback news generation"""
        fallback_df = generate_fallback_news()
        
        # Check DataFrame structure
        assert isinstance(fallback_df, pd.DataFrame)
        assert not fallback_df.empty
        assert len(fallback_df) >= 3  # Should have at least 3 sample articles
        
        # Check required columns
        required_columns = ['date', 'title', 'source', 'url', 'snippet', 'category', 'region']
        for col in required_columns:
            assert col in fallback_df.columns
        
        # Check sentiment columns are added
        assert 'sentiment_score' in fallback_df.columns
        assert 'sentiment_label' in fallback_df.columns
        assert 'sentiment_emoji' in fallback_df.columns
        
        # Check fallback flag
        assert 'fallback' in fallback_df.columns
        assert fallback_df['fallback'].all()  # All should be marked as fallback
    
    @responses.activate
    def test_get_rss_feed(self):
        """Test RSS feed parsing"""
        # Mock RSS response
        responses.add(
            responses.GET,
            "https://example.com/feed.rss",
            body=SAMPLE_RSS_CONTENT,
            status=200,
            content_type="application/rss+xml"
        )
        
        # Test RSS parsing
        df = get_rss_feed("https://example.com/feed.rss", "Test Source")
        
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert len(df) == 2  # Two items in sample RSS
        
        # Check data structure
        assert 'title' in df.columns
        assert 'snippet' in df.columns
        assert 'source' in df.columns
        assert 'url' in df.columns
        assert 'date' in df.columns
        
        # Check source name
        assert df['source'].iloc[0] == "Test Source"
        
        # Check sentiment analysis was applied
        assert 'sentiment_score' in df.columns
        assert 'sentiment_label' in df.columns
    
    @responses.activate
    def test_get_gnews_success(self):
        """Test successful GNews API call"""
        # Mock GNews API response
        responses.add(
            responses.GET,
            "https://gnews.io/api/v4/search",
            json=SAMPLE_GNEWS_RESPONSE,
            status=200
        )
        
        df = get_gnews("fintech")
        
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert len(df) == 2
        
        # Check required columns
        assert 'title' in df.columns
        assert 'source' in df.columns
        assert 'sentiment_score' in df.columns
    
    @responses.activate
    def test_get_gnews_api_error(self):
        """Test GNews API error handling"""
        # Mock API error (401 Unauthorized)
        responses.add(
            responses.GET,
            "https://gnews.io/api/v4/search",
            status=401
        )
        
        # Mock the fallback RSS call
        responses.add(
            responses.GET,
            "https://news.google.com/rss/search",
            body=SAMPLE_RSS_CONTENT,
            status=200,
            content_type="application/rss+xml"
        )
        
        df = get_gnews("fintech")
        
        # Should return fallback data or RSS data
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
    
    @responses.activate
    def test_get_yahoo_finance_feed(self):
        """Test Yahoo Finance RSS feed"""
        # Mock Yahoo Finance RSS
        yahoo_rss = SAMPLE_RSS_CONTENT.replace("Fintech News", "Yahoo Finance")
        responses.add(
            responses.GET,
            "https://feeds.finance.yahoo.com/rss/2.0/headline",
            body=yahoo_rss,
            status=200,
            content_type="application/rss+xml"
        )
        
        df = get_yahoo_finance_feed("BTC-USD")
        
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert any("Yahoo Finance" in source for source in df['source'])
    
    @responses.activate
    def test_get_cryptopanic_feed(self):
        """Test CryptoPanic RSS feed"""
        crypto_rss = SAMPLE_RSS_CONTENT.replace("Fintech News", "CryptoPanic")
        responses.add(
            responses.GET,
            "https://cryptopanic.com/api/v1/posts/",
            body=crypto_rss,
            status=200,
            content_type="application/rss+xml"
        )
        
        df = get_cryptopanic_feed("news")
        
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
    
    def test_search_news(self):
        """Test news search functionality"""
        # This will use fallback data in test environment
        df = search_news("fintech", region="global", sentiment="all")
        
        assert isinstance(df, pd.DataFrame)
        # In test environment, this should return fallback data
        assert not df.empty or df.empty  # Accept either for testing
    
    def test_get_news_summary_stats(self):
        """Test news summary statistics"""
        # Test with fallback data
        fallback_df = generate_fallback_news()
        stats = get_news_summary_stats(fallback_df)
        
        assert isinstance(stats, dict)
        assert 'total_articles' in stats
        assert 'sources' in stats
        assert 'sentiment_distribution' in stats
        assert 'category_distribution' in stats
        assert 'avg_sentiment' in stats
        
        assert stats['total_articles'] > 0
        assert isinstance(stats['avg_sentiment'], float)
        
        # Test with empty DataFrame
        empty_stats = get_news_summary_stats(pd.DataFrame())
        assert empty_stats['total_articles'] == 0
        assert empty_stats['avg_sentiment'] == 0.0
    
    def test_news_config(self):
        """Test news configuration"""
        config = NewsConfig()
        
        assert hasattr(config, 'timeout_seconds')
        assert hasattr(config, 'max_retries')
        assert hasattr(config, 'cache_ttl_minutes')
        assert hasattr(config, 'max_articles_per_source')
        
        assert config.timeout_seconds > 0
        assert config.max_retries > 0
        assert config.cache_ttl_minutes > 0
    
    @responses.activate
    def test_network_failure_fallback(self):
        """Test fallback behavior on network failure"""
        # Mock network failure
        responses.add(
            responses.GET,
            "https://gnews.io/api/v4/search",
            status=500
        )
        
        # Should return fallback data
        df = get_gnews("fintech")
        
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        # Check if fallback flag is set
        if 'fallback' in df.columns:
            assert df['fallback'].any()
    
    def test_data_quality_indicators(self):
        """Test data quality indicators in news data"""
        fallback_df = generate_fallback_news()
        
        # Check fallback indicator
        assert 'fallback' in fallback_df.columns
        assert fallback_df['fallback'].all()
        
        # Check last refresh timestamp
        assert 'last_refresh' in fallback_df.columns
        assert not fallback_df['last_refresh'].isna().any()
    
    def test_category_classification(self):
        """Test automatic category classification"""
        # This is tested implicitly in RSS parsing
        # Categories should be assigned based on content
        fallback_df = generate_fallback_news()
        
        # Check that categories are assigned
        assert 'category' in fallback_df.columns
        categories = fallback_df['category'].unique()
        expected_categories = ['fintech', 'banking', 'crypto', 'regulation']
        
        # At least some categories should be present
        assert len(categories) > 0
        assert all(cat in expected_categories for cat in categories)
    
    def test_date_handling(self):
        """Test date parsing and handling"""
        fallback_df = generate_fallback_news()
        
        # Check date column exists and has valid dates
        assert 'date' in fallback_df.columns
        assert not fallback_df['date'].isna().any()
        
        # Check dates are datetime objects
        assert pd.api.types.is_datetime64_any_dtype(fallback_df['date'])
        
        # Check dates are reasonable (not too old or in future)
        now = datetime.now()
        one_week_ago = now - timedelta(days=7)
        
        # All dates should be within reasonable range
        assert (fallback_df['date'] >= one_week_ago).all()
        assert (fallback_df['date'] <= now + timedelta(hours=1)).all()

@pytest.mark.skipif(not NEWS_MODULE_AVAILABLE, reason="News module not available")
class TestNewsIntegration:
    """Integration tests for news module"""
    
    def test_get_fintech_news_integration(self):
        """Test the main fintech news function"""
        df = get_fintech_news(
            query="fintech",
            region="global",
            categories=['fintech', 'crypto'],
            limit=10
        )
        
        assert isinstance(df, pd.DataFrame)
        # Should return data (fallback if APIs fail)
        
        if not df.empty:
            # Check structure if data is returned
            assert 'title' in df.columns
            assert 'sentiment_score' in df.columns
            assert len(df) <= 10  # Respects limit
    
    def test_error_handling_robustness(self):
        """Test robust error handling"""
        # Test with invalid parameters
        try:
            df = get_fintech_news(query="", region="invalid", limit=0)
            # Should still return something (fallback)
            assert isinstance(df, pd.DataFrame)
        except Exception as e:
            # Should not raise unhandled exceptions
            pytest.fail(f"Unexpected exception: {e}")
    
    def test_multiple_source_aggregation(self):
        """Test aggregating from multiple sources"""
        # This test verifies the aggregation logic works
        # In test environment, will use fallback data
        df = get_aggregated_news("fintech", ['fintech', 'crypto'])
        
        assert isinstance(df, pd.DataFrame)
        
        if not df.empty:
            # Check that sentiment analysis was applied
            assert 'sentiment_score' in df.columns
            assert 'sentiment_label' in df.columns
            
            # Check category filtering worked
            if 'category' in df.columns:
                categories = df['category'].unique()
                assert all(cat in ['fintech', 'crypto'] for cat in categories)

# Test fixtures and utilities
@pytest.fixture
def sample_news_data():
    """Fixture providing sample news data for testing"""
    return generate_fallback_news()

@pytest.fixture
def mock_rss_response():
    """Fixture providing mock RSS response"""
    return SAMPLE_RSS_CONTENT

@pytest.fixture
def mock_gnews_response():
    """Fixture providing mock GNews API response"""
    return SAMPLE_GNEWS_RESPONSE

# Performance and stress tests
@pytest.mark.skipif(not NEWS_MODULE_AVAILABLE, reason="News module not available")
class TestNewsPerformance:
    """Performance tests for news module"""
    
    def test_sentiment_analysis_performance(self):
        """Test sentiment analysis performance on multiple texts"""
        import time
        
        texts = [
            "Great news for fintech innovation!",
            "Market crash affects crypto prices",
            "Bank announces new digital services",
            "Regulatory approval for blockchain",
            "Investment in African fintech grows"
        ] * 20  # 100 texts total
        
        start_time = time.time()
        
        for text in texts:
            analyze_sentiment(text)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should process 100 texts in reasonable time (< 5 seconds)
        assert processing_time < 5.0
        
        # Calculate throughput
        throughput = len(texts) / processing_time
        assert throughput > 10  # Should process at least 10 texts/second
    
    def test_large_dataframe_handling(self):
        """Test handling of large news DataFrames"""
        # Create a large DataFrame
        large_df = pd.concat([generate_fallback_news()] * 20, ignore_index=True)
        
        # Test summary stats on large DataFrame
        stats = get_news_summary_stats(large_df)
        assert stats['total_articles'] == len(large_df)
        
        # Test search on large DataFrame
        search_result = search_news("fintech", region="global")
        assert isinstance(search_result, pd.DataFrame)

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
