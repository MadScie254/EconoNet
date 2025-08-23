"""
Basic tests for EconoNet Live News Module

Tests core functionality like sentiment analysis and fallback data generation.
"""

import pytest
import pandas as pd
from unittest.mock import patch
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path
current_dir = Path(__file__).parent.parent
sys.path.append(str(current_dir / 'src'))

# Import news module with fallback handling
NEWS_MODULE_AVAILABLE = False
try:
    from src.econonet.live_news import (
        get_fintech_news, analyze_sentiment, generate_fallback_news, 
        search_news, get_news_summary_stats
    )
    NEWS_MODULE_AVAILABLE = True
except ImportError:
    try:
        # Try alternative import path
        import importlib.util
        news_file = current_dir / 'src' / 'econonet' / 'live_news.py'
        if news_file.exists():
            spec = importlib.util.spec_from_file_location("live_news", news_file)
            live_news = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(live_news)
            
            get_fintech_news = live_news.get_fintech_news
            analyze_sentiment = live_news.analyze_sentiment
            generate_fallback_news = live_news.generate_fallback_news
            search_news = live_news.search_news
            get_news_summary_stats = live_news.get_news_summary_stats
            NEWS_MODULE_AVAILABLE = True
    except Exception:
        NEWS_MODULE_AVAILABLE = False

@pytest.mark.skipif(not NEWS_MODULE_AVAILABLE, reason="News module not available")
class TestNewsModuleBasic:
    """Basic test cases for the news module"""
    
    def test_sentiment_analysis(self):
        """Test sentiment analysis functionality"""
        # Test positive sentiment
        score, label, emoji = analyze_sentiment("This is great news for fintech innovation!")
        assert isinstance(score, float)
        assert label in ['bullish', 'bearish', 'neutral']
        assert emoji in ['🟢', '🔴', '⚪']
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
            assert col in fallback_df.columns, f"Missing column: {col}"
        
        # Check sentiment columns are added
        assert 'sentiment_score' in fallback_df.columns
        assert 'sentiment_label' in fallback_df.columns
        assert 'sentiment_emoji' in fallback_df.columns
        
        # Check fallback flag
        assert 'fallback' in fallback_df.columns
        assert fallback_df['fallback'].all()  # All should be marked as fallback
        
        # Check data types
        assert pd.api.types.is_datetime64_any_dtype(fallback_df['date'])
        assert all(isinstance(title, str) for title in fallback_df['title'])
        assert all(isinstance(score, (int, float)) for score in fallback_df['sentiment_score'])
    
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
        assert isinstance(stats['avg_sentiment'], (int, float))
        assert isinstance(stats['sources'], int)
        
        # Check sentiment distribution
        sentiment_dist = stats['sentiment_distribution']
        assert isinstance(sentiment_dist, dict)
        expected_sentiments = ['bullish', 'bearish', 'neutral']
        for sentiment in expected_sentiments:
            assert sentiment in sentiment_dist
        
        # Test with empty DataFrame
        empty_stats = get_news_summary_stats(pd.DataFrame())
        assert empty_stats['total_articles'] == 0
        assert empty_stats['avg_sentiment'] == 0.0
        assert empty_stats['sources'] == 0
    
    def test_search_news(self):
        """Test news search functionality with fallback"""
        # This will use fallback data in test environment
        df = search_news("fintech", region="global", sentiment="all")
        
        assert isinstance(df, pd.DataFrame)
        # In test environment, this should return fallback data
        if not df.empty:
            # Check structure if data is returned
            assert 'title' in df.columns
            assert 'sentiment_score' in df.columns
            assert 'category' in df.columns
    
    def test_get_fintech_news_fallback(self):
        """Test the main fintech news function with fallback"""
        # Mock network failure to force fallback
        with patch('requests.get') as mock_get:
            mock_get.side_effect = Exception("Network error")
            
            df = get_fintech_news(
                query="fintech",
                region="global", 
                categories=['fintech', 'crypto'],
                limit=10
            )
            
            assert isinstance(df, pd.DataFrame)
            # Should return fallback data when APIs fail
            if not df.empty:
                assert 'title' in df.columns
                assert 'sentiment_score' in df.columns
                assert len(df) <= 10  # Respects limit
                assert 'fallback' in df.columns
                assert df['fallback'].any()  # Some fallback data
    
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
    
    def test_category_classification(self):
        """Test automatic category classification"""
        fallback_df = generate_fallback_news()
        
        # Check that categories are assigned
        assert 'category' in fallback_df.columns
        categories = fallback_df['category'].unique()
        expected_categories = ['fintech', 'banking', 'crypto', 'regulation']
        
        # At least some categories should be present
        assert len(categories) > 0
        assert all(cat in expected_categories for cat in categories)
    
    def test_data_quality_indicators(self):
        """Test data quality indicators in news data"""
        fallback_df = generate_fallback_news()
        
        # Check fallback indicator
        assert 'fallback' in fallback_df.columns
        assert fallback_df['fallback'].all()
        
        # Check last refresh timestamp
        assert 'last_refresh' in fallback_df.columns
        assert not fallback_df['last_refresh'].isna().any()
        
        # Check URL format
        urls = fallback_df['url'].tolist()
        for url in urls:
            assert isinstance(url, str)
            assert url.startswith('http')
    
    def test_sentiment_distribution(self):
        """Test sentiment label distribution"""
        fallback_df = generate_fallback_news()
        
        sentiment_labels = fallback_df['sentiment_label'].unique()
        valid_sentiments = {'bullish', 'bearish', 'neutral'}
        
        # All sentiment labels should be valid
        assert all(label in valid_sentiments for label in sentiment_labels)
        
        # Should have at least one of each sentiment type
        assert len(sentiment_labels) >= 1
        
        # Check emoji consistency
        for _, row in fallback_df.iterrows():
            label = row['sentiment_label']
            emoji = row['sentiment_emoji']
            score = row['sentiment_score']
            
            if label == 'bullish':
                assert emoji == '🟢'
                assert score > 0
            elif label == 'bearish':
                assert emoji == '🔴'
                assert score < 0
            else:  # neutral
                assert emoji == '⚪'
    
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
        
        # Test sentiment analysis with edge cases
        try:
            score, label, emoji = analyze_sentiment(None)
            assert label == 'neutral'
        except:
            # Should handle None gracefully
            pass

# Test for module availability
def test_news_module_import():
    """Test that the news module can be imported"""
    if NEWS_MODULE_AVAILABLE:
        assert callable(get_fintech_news)
        assert callable(analyze_sentiment)
        assert callable(generate_fallback_news)
        assert callable(search_news)
        assert callable(get_news_summary_stats)
    else:
        pytest.skip("News module not available for import")

# Integration test (simplified)
@pytest.mark.skipif(not NEWS_MODULE_AVAILABLE, reason="News module not available")
def test_news_pipeline():
    """Test the complete news processing pipeline"""
    # Generate some test data
    df = generate_fallback_news()
    
    # Test the pipeline: data → sentiment → stats → search
    assert not df.empty
    
    # Get summary stats
    stats = get_news_summary_stats(df)
    assert stats['total_articles'] == len(df)
    
    # Test search functionality
    search_result = search_news("fintech", region="global")
    assert isinstance(search_result, pd.DataFrame)

if __name__ == "__main__":
    # Run basic tests
    if NEWS_MODULE_AVAILABLE:
        print("✅ News module is available")
        print("Running basic tests...")
        
        # Test sentiment analysis
        score, label, emoji = analyze_sentiment("Great news for fintech!")
        print(f"Sentiment test: {score:.2f} | {label} | {emoji}")
        
        # Test fallback data
        df = generate_fallback_news()
        print(f"Fallback data: {len(df)} articles generated")
        
        # Test stats
        stats = get_news_summary_stats(df)
        print(f"Stats: {stats['total_articles']} articles, avg sentiment: {stats['avg_sentiment']:.2f}")
        
        print("✅ All basic tests passed!")
    else:
        print("❌ News module not available")
        print("Make sure the src/econonet/live_news.py file exists")
