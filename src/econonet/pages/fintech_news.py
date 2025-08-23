"""
Fintech News & Insights Dashboard Page

Interactive Streamlit page for browsing fintech, crypto, and finance news
with sentiment analysis, filtering, and visualization capabilities.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
from pathlib import Path

# Add src to path for imports
current_dir = Path(__file__).parent.parent.parent.parent
sys.path.append(str(current_dir / 'src'))

try:
    from econonet.live_news import (
        get_fintech_news, search_news, get_news_summary_stats,
        get_gnews, get_yahoo_finance_feed, get_cryptopanic_feed,
        get_rss_feed, get_aggregated_news
    )
    from econonet.visual.news_cards import (
        create_news_grid, create_sentiment_timeline, create_sentiment_radar,
        create_category_distribution, create_source_activity, 
        create_news_summary_metrics, display_news_metrics, inject_news_css
    )
    from econonet.visual.provenance_footer import create_provenance_footer
    NEWS_AVAILABLE = True
except ImportError as e:
    st.error(f"News module import failed: {e}")
    NEWS_AVAILABLE = False

def create_youtube_embed(video_id: str, title: str = "Fintech Video") -> str:
    """Create YouTube embed HTML"""
    return f"""
    <div style="margin: 1rem 0;">
        <h4 style="color: white; margin-bottom: 0.5rem;">{title}</h4>
        <iframe 
            width="100%" 
            height="315" 
            src="https://www.youtube.com/embed/{video_id}" 
            frameborder="0" 
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
            allowfullscreen
            style="border-radius: 10px;">
        </iframe>
    </div>
    """

def get_fintech_videos() -> list:
    """Get list of fintech-related video IDs (manually curated for demo)"""
    return [
        {"id": "dQw4w9WgXcQ", "title": "🚀 The Future of African Fintech"},
        {"id": "dQw4w9WgXcQ", "title": "💰 Central Bank Digital Currencies Explained"},
        {"id": "dQw4w9WgXcQ", "title": "🌍 Mobile Money Revolution in Africa"}
    ]

def main():
    """Main function for the Fintech News & Insights page"""
    
    # Page configuration
    st.set_page_config(
        page_title="EconoNet - Fintech News & Insights",
        page_icon="📰",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Inject custom CSS
    inject_news_css()
    
    # Custom CSS for the page
    st.markdown("""
    <style>
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 15px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }
        
        .filter-section {
            background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 15px;
            padding: 1.5rem;
            margin-bottom: 2rem;
        }
        
        .video-section {
            background: linear-gradient(135deg, rgba(0,0,0,0.8), rgba(102,126,234,0.2));
            border-radius: 15px;
            padding: 1.5rem;
            margin: 1rem 0;
        }
        
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        
        .stTabs [data-baseweb="tab"] {
            background: rgba(255,255,255,0.1);
            border-radius: 10px;
            color: white;
            border: 1px solid rgba(255,255,255,0.2);
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(45deg, #667eea, #764ba2);
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📰 Fintech News & Insights</h1>
        <p>Real-time financial technology news with sentiment analysis and trend visualization</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar Controls
    with st.sidebar:
        st.markdown("### 🎛️ News Filters")
        
        # Category filter
        category_options = ['All', 'Fintech', 'Banking', 'Crypto', 'Regulation', 'Africa']
        selected_category = st.selectbox(
            "📂 Category",
            options=category_options,
            index=0,
            help="Filter news by category"
        )
        
        # Region filter
        region_options = ['Global', 'Africa', 'Kenya', 'Nigeria', 'South Africa']
        selected_region = st.selectbox(
            "🌍 Region",
            options=region_options,
            index=0,
            help="Filter news by geographical region"
        )
        
        # Sentiment filter
        sentiment_options = ['All', 'Bullish', 'Bearish', 'Neutral']
        selected_sentiment = st.selectbox(
            "😊 Sentiment",
            options=sentiment_options,
            index=0,
            help="Filter news by sentiment analysis"
        )
        
        # Search query
        search_query = st.text_input(
            "🔍 Search Keywords",
            value="fintech",
            help="Enter keywords to search for specific topics"
        )
        
        # Sort options
        sort_options = ['Date (Newest)', 'Date (Oldest)', 'Source', 'Sentiment']
        selected_sort = st.selectbox(
            "🔄 Sort By",
            options=sort_options,
            index=0,
            help="Choose sorting criteria"
        )
        
        # Article limit
        article_limit = st.slider(
            "📊 Max Articles",
            min_value=10,
            max_value=100,
            value=30,
            step=10,
            help="Maximum number of articles to display"
        )
        
        st.markdown("---")
        
        # Refresh button
        if st.button("🔄 Refresh News", type="primary"):
            st.cache_data.clear()
            st.rerun()
        
        # Data source info
        st.markdown("### 📡 Data Sources")
        st.markdown("""
        - 📰 Google News (RSS)
        - 💰 Yahoo Finance
        - 🪙 CryptoPanic
        - 📊 Reuters Business
        - 🌐 BBC Business
        """)
    
    # Check if news module is available
    if not NEWS_AVAILABLE:
        st.error("📡 News module not available. Please check your installation.")
        st.info("Using fallback mode with sample data.")
        return
    
    # Main content area
    try:
        # Get news data
        with st.spinner("📡 Fetching latest fintech news..."):
            # Convert filters to API parameters
            region_param = selected_region.lower() if selected_region != 'Global' else 'global'
            category_param = [selected_category.lower()] if selected_category != 'All' else None
            
            # Fetch news
            news_df = get_fintech_news(
                query=search_query,
                region=region_param,
                categories=category_param,
                limit=article_limit
            )
            
            # Apply sentiment filter
            if selected_sentiment != 'All':
                sentiment_param = selected_sentiment.lower()
                news_df = news_df[news_df['sentiment_label'] == sentiment_param]
            
            # Apply sorting
            if selected_sort == 'Date (Newest)':
                news_df = news_df.sort_values('date', ascending=False)
            elif selected_sort == 'Date (Oldest)':
                news_df = news_df.sort_values('date', ascending=True)
            elif selected_sort == 'Source':
                news_df = news_df.sort_values('source')
            elif selected_sort == 'Sentiment':
                news_df = news_df.sort_values('sentiment_score', ascending=False)
        
        # Display metrics
        if not news_df.empty:
            metrics = create_news_summary_metrics(news_df)
            display_news_metrics(metrics)
        
        st.markdown("---")
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📰 News Feed", "📈 Analytics", "🎥 Videos", "⚙️ Sources"])
        
        with tab1:
            st.markdown("### 📰 Latest News Articles")
            
            if news_df.empty:
                st.info("📭 No news articles found with current filters. Try adjusting your search criteria.")
            else:
                # Display article count and filters
                st.markdown(f"**Found {len(news_df)} articles** matching your criteria")
                
                # News grid
                articles_list = news_df.to_dict('records')
                create_news_grid(articles_list, columns=2)
        
        with tab2:
            st.markdown("### 📈 News Analytics & Insights")
            
            if news_df.empty:
                st.info("📊 No data available for analytics. Adjust your filters to see insights.")
            else:
                # Analytics visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    # Sentiment timeline
                    timeline_fig = create_sentiment_timeline(news_df)
                    st.plotly_chart(timeline_fig, use_container_width=True)
                    
                    # Category distribution
                    category_fig = create_category_distribution(news_df)
                    st.plotly_chart(category_fig, use_container_width=True)
                
                with col2:
                    # Sentiment radar
                    radar_fig = create_sentiment_radar(news_df)
                    st.plotly_chart(radar_fig, use_container_width=True)
                    
                    # Source activity
                    source_fig = create_source_activity(news_df)
                    st.plotly_chart(source_fig, use_container_width=True)
                
                # Summary statistics
                st.markdown("### 📊 Summary Statistics")
                stats = get_news_summary_stats(news_df)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Sources", stats['sources'])
                with col2:
                    st.metric("Avg Sentiment", f"{stats['avg_sentiment']:.3f}")
                with col3:
                    st.metric("Fallback Ratio", f"{stats['fallback_ratio']:.1%}")
        
        with tab3:
            st.markdown("### 🎥 Fintech Video Insights")
            
            # Video section
            st.markdown("""
            <div class="video-section">
                <h4 style="color: white; margin-bottom: 1rem;">🎬 Featured Fintech Content</h4>
                <p style="color: rgba(255,255,255,0.8);">
                    Explore the latest video content about fintech trends, digital banking, 
                    and financial innovation across Africa and globally.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Display videos
            videos = get_fintech_videos()
            for i, video in enumerate(videos[:3]):  # Limit to 3 videos
                with st.expander(f"🎥 {video['title']}", expanded=(i == 0)):
                    st.markdown(
                        create_youtube_embed(video['id'], video['title']),
                        unsafe_allow_html=True
                    )
            
            # Video search and recommendations
            st.markdown("### 🔍 Video Search")
            video_query = st.text_input(
                "Search for fintech videos",
                placeholder="e.g., 'mobile payments africa', 'blockchain banking'"
            )
            
            if video_query:
                st.info(f"🎬 Searching for videos about: '{video_query}'")
                st.markdown("*Video search integration would connect to YouTube Data API or similar service.*")
        
        with tab4:
            st.markdown("### ⚙️ Data Sources & Configuration")
            
            # Source status
            st.markdown("#### 📡 Live Data Sources")
            
            # Test individual sources
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🧪 Test Google News"):
                    with st.spinner("Testing Google News API..."):
                        try:
                            test_df = get_gnews("fintech")
                            if not test_df.empty:
                                st.success(f"✅ Google News: {len(test_df)} articles retrieved")
                            else:
                                st.warning("⚠️ Google News: No articles found")
                        except Exception as e:
                            st.error(f"❌ Google News: {str(e)}")
                
                if st.button("🧪 Test Yahoo Finance"):
                    with st.spinner("Testing Yahoo Finance RSS..."):
                        try:
                            test_df = get_yahoo_finance_feed("BTC-USD")
                            if not test_df.empty:
                                st.success(f"✅ Yahoo Finance: {len(test_df)} articles retrieved")
                            else:
                                st.warning("⚠️ Yahoo Finance: No articles found")
                        except Exception as e:
                            st.error(f"❌ Yahoo Finance: {str(e)}")
            
            with col2:
                if st.button("🧪 Test CryptoPanic"):
                    with st.spinner("Testing CryptoPanic RSS..."):
                        try:
                            test_df = get_cryptopanic_feed("news")
                            if not test_df.empty:
                                st.success(f"✅ CryptoPanic: {len(test_df)} articles retrieved")
                            else:
                                st.warning("⚠️ CryptoPanic: No articles found")
                        except Exception as e:
                            st.error(f"❌ CryptoPanic: {str(e)}")
                
                if st.button("🧪 Test Custom RSS"):
                    with st.spinner("Testing Reuters RSS..."):
                        try:
                            test_df = get_rss_feed("https://www.reuters.com/business/finance/rss", "Reuters")
                            if not test_df.empty:
                                st.success(f"✅ Reuters RSS: {len(test_df)} articles retrieved")
                            else:
                                st.warning("⚠️ Reuters RSS: No articles found")
                        except Exception as e:
                            st.error(f"❌ Reuters RSS: {str(e)}")
            
            # Configuration info
            st.markdown("#### ⚙️ System Configuration")
            config_info = {
                "Cache TTL": "15 minutes",
                "Max Articles per Source": "50",
                "Request Timeout": "30 seconds",
                "Retry Attempts": "3",
                "Sentiment Analysis": "TextBlob (VADER)",
                "Fallback Mode": "Enabled"
            }
            
            for key, value in config_info.items():
                st.text(f"• {key}: {value}")
            
            # RSS feed manager
            st.markdown("#### 📡 Custom RSS Feeds")
            custom_rss = st.text_input(
                "Add Custom RSS Feed URL",
                placeholder="https://example.com/feed.rss"
            )
            
            if custom_rss and st.button("Test Custom Feed"):
                with st.spinner("Testing custom RSS feed..."):
                    try:
                        test_df = get_rss_feed(custom_rss, "Custom Feed")
                        if not test_df.empty:
                            st.success(f"✅ Custom Feed: {len(test_df)} articles retrieved")
                            st.dataframe(test_df[['title', 'source', 'date']].head())
                        else:
                            st.warning("⚠️ Custom Feed: No articles found")
                    except Exception as e:
                        st.error(f"❌ Custom Feed Error: {str(e)}")
        
        # Footer with provenance
        st.markdown("---")
        if not news_df.empty:
            sources = news_df['source'].unique().tolist()
            last_update = news_df['date'].max().strftime('%Y-%m-%d %H:%M')
            create_provenance_footer(
                f"News data from {len(sources)} sources", 
                f"Last updated: {last_update}",
                data_quality="live" if not news_df.get('fallback', pd.Series([True])).any() else "mixed"
            )
        
    except Exception as e:
        st.error(f"An error occurred while loading news data: {str(e)}")
        st.info("🔄 Please try refreshing the page or check your internet connection.")
        
        # Show fallback message
        st.markdown("""
        ### 📡 Fallback Mode
        
        The news module is currently using fallback data. This could be due to:
        - Network connectivity issues
        - API rate limiting
        - Temporary service unavailability
        
        Please try again in a few minutes for live data.
        """)

if __name__ == "__main__":
    main()
