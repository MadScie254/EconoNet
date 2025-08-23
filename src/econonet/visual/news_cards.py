"""
EconoNet News Card Visual Components

Renders styled news cards with sentiment badges and interactive elements
for the Fintech News & Insights dashboard.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import numpy as np

def create_news_card(article: Dict[str, Any], card_id: str = "") -> None:
    """
    Create a styled news card for a single article
    
    Args:
        article: Dictionary with article data
        card_id: Unique identifier for the card
    """
    # Extract article data
    title = article.get('title', 'No Title')
    snippet = article.get('snippet', 'No description available')
    source = article.get('source', 'Unknown Source')
    url = article.get('url', '#')
    date_str = article.get('date', datetime.now()).strftime('%Y-%m-%d %H:%M') if isinstance(article.get('date'), datetime) else str(article.get('date', ''))
    sentiment_emoji = article.get('sentiment_emoji', '⚪')
    sentiment_label = article.get('sentiment_label', 'neutral')
    category = article.get('category', 'fintech').title()
    fallback = article.get('fallback', False)
    
    # Determine sentiment color
    sentiment_colors = {
        'bullish': '#00ff88',
        'bearish': '#ff4757',
        'neutral': '#747d8c'
    }
    sentiment_color = sentiment_colors.get(sentiment_label, '#747d8c')
    
    # Create card HTML with enhanced styling
    card_html = f"""
    <div class="news-card" style="
        background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    ">
        <!-- Sentiment Badge -->
        <div style="
            position: absolute;
            top: 15px;
            right: 15px;
            background: {sentiment_color};
            color: white;
            padding: 0.3rem 0.8rem;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 0.3rem;
        ">
            {sentiment_emoji} {sentiment_label.title()}
        </div>
        
        <!-- Category Badge -->
        <div style="
            display: inline-block;
            background: rgba(102,126,234,0.2);
            color: #667eea;
            padding: 0.2rem 0.6rem;
            border-radius: 12px;
            font-size: 0.7rem;
            font-weight: 500;
            margin-bottom: 0.8rem;
        ">
            {category}
        </div>
        
        <!-- Fallback indicator -->
        {'<div style="color: #ffa502; font-size: 0.7rem; margin-bottom: 0.5rem;">📡 Fallback Data</div>' if fallback else ''}
        
        <!-- Article Title -->
        <h3 style="
            color: white;
            font-size: 1.2rem;
            font-weight: 600;
            margin: 0.5rem 0;
            line-height: 1.4;
            padding-right: 6rem;
        ">{title}</h3>
        
        <!-- Article Snippet -->
        <p style="
            color: rgba(255,255,255,0.8);
            font-size: 0.95rem;
            line-height: 1.6;
            margin: 1rem 0;
            display: -webkit-box;
            -webkit-line-clamp: 3;
            -webkit-box-orient: vertical;
            overflow: hidden;
        ">{snippet}</p>
        
        <!-- Article Metadata -->
        <div style="
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 1.5rem;
            padding-top: 1rem;
            border-top: 1px solid rgba(255,255,255,0.1);
        ">
            <div style="
                display: flex;
                flex-direction: column;
                gap: 0.2rem;
            ">
                <span style="
                    color: rgba(255,255,255,0.7);
                    font-size: 0.8rem;
                    font-weight: 500;
                ">{source}</span>
                <span style="
                    color: rgba(255,255,255,0.5);
                    font-size: 0.75rem;
                ">{date_str}</span>
            </div>
            
            <a href="{url}" target="_blank" style="
                background: linear-gradient(45deg, #667eea, #764ba2);
                color: white;
                padding: 0.6rem 1.2rem;
                border-radius: 25px;
                text-decoration: none;
                font-size: 0.85rem;
                font-weight: 500;
                transition: all 0.3s ease;
                border: none;
                cursor: pointer;
            ">
                Read More →
            </a>
        </div>
    </div>
    """
    
    # Display the card
    st.markdown(card_html, unsafe_allow_html=True)

def create_news_grid(articles: List[Dict[str, Any]], columns: int = 2) -> None:
    """
    Create a grid layout of news cards
    
    Args:
        articles: List of article dictionaries
        columns: Number of columns in the grid
    """
    if not articles:
        st.info("📰 No news articles available at the moment.")
        return
    
    # Create columns
    cols = st.columns(columns)
    
    for i, article in enumerate(articles):
        with cols[i % columns]:
            create_news_card(article, f"card_{i}")

def create_sentiment_timeline(news_df: pd.DataFrame) -> go.Figure:
    """
    Create a timeline visualization of news sentiment
    
    Args:
        news_df: DataFrame with news data
    
    Returns:
        Plotly figure
    """
    if news_df.empty:
        # Create empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No news data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="white")
        )
        fig.update_layout(
            title="📈 News Sentiment Timeline",
            template='plotly_dark',
            height=400,
            font=dict(color='white')
        )
        return fig
    
    # Prepare data
    df = news_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # Create color mapping
    color_map = {
        'bullish': '#00ff88',
        'bearish': '#ff4757',
        'neutral': '#747d8c'
    }
    
    df['color'] = df['sentiment_label'].map(color_map)
    
    # Create figure
    fig = go.Figure()
    
    # Add scatter plot for each sentiment
    for sentiment in ['bullish', 'bearish', 'neutral']:
        sentiment_data = df[df['sentiment_label'] == sentiment]
        if not sentiment_data.empty:
            fig.add_trace(go.Scatter(
                x=sentiment_data['date'],
                y=sentiment_data['sentiment_score'],
                mode='markers',
                marker=dict(
                    color=color_map[sentiment],
                    size=10,
                    line=dict(color='white', width=1)
                ),
                name=f"{sentiment.title()} ({len(sentiment_data)})",
                text=sentiment_data['title'],
                hovertemplate='<b>%{text}</b><br>' +
                             'Date: %{x}<br>' +
                             'Sentiment: %{y:.2f}<br>' +
                             '<extra></extra>'
            ))
    
    # Add trend line
    if len(df) > 1:
        # Create moving average
        df['sentiment_ma'] = df['sentiment_score'].rolling(window=min(5, len(df))).mean()
        
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['sentiment_ma'],
            mode='lines',
            line=dict(color='#4facfe', width=2),
            name='Sentiment Trend',
            hovertemplate='Trend: %{y:.2f}<br>Date: %{x}<extra></extra>'
        ))
    
    fig.update_layout(
        title='📈 News Sentiment Timeline',
        xaxis_title='Date',
        yaxis_title='Sentiment Score',
        template='plotly_dark',
        height=400,
        font=dict(color='white'),
        hovermode='closest',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_sentiment_radar(news_df: pd.DataFrame) -> go.Figure:
    """
    Create a radar chart showing sentiment distribution by category
    
    Args:
        news_df: DataFrame with news data
    
    Returns:
        Plotly figure
    """
    if news_df.empty:
        # Create empty radar chart
        fig = go.Figure()
        fig.add_annotation(
            text="No data for sentiment analysis",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="white")
        )
        fig.update_layout(
            title="🎯 Sentiment Radar by Category",
            template='plotly_dark',
            height=400,
            font=dict(color='white')
        )
        return fig
    
    # Calculate sentiment scores by category
    category_sentiment = news_df.groupby('category').agg({
        'sentiment_score': 'mean',
        'title': 'count'
    }).reset_index()
    category_sentiment.columns = ['category', 'avg_sentiment', 'article_count']
    
    # Normalize sentiment scores to 0-100 scale for radar chart
    category_sentiment['sentiment_normalized'] = ((category_sentiment['avg_sentiment'] + 1) * 50).clip(0, 100)
    
    # Create radar chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=category_sentiment['sentiment_normalized'],
        theta=category_sentiment['category'],
        fill='toself',
        line=dict(color='#4facfe', width=2),
        fillcolor='rgba(79, 172, 254, 0.3)',
        name='Sentiment Score',
        text=category_sentiment['article_count'],
        hovertemplate='<b>%{theta}</b><br>' +
                     'Sentiment: %{r:.1f}/100<br>' +
                     'Articles: %{text}<br>' +
                     '<extra></extra>'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                gridcolor='rgba(255,255,255,0.2)',
                gridwidth=1,
                tickcolor='white'
            ),
            angularaxis=dict(
                gridcolor='rgba(255,255,255,0.2)',
                gridwidth=1,
                tickcolor='white'
            )
        ),
        title='🎯 Sentiment Radar by Category',
        template='plotly_dark',
        height=400,
        font=dict(color='white'),
        showlegend=False
    )
    
    return fig

def create_category_distribution(news_df: pd.DataFrame) -> go.Figure:
    """
    Create a donut chart showing article distribution by category
    
    Args:
        news_df: DataFrame with news data
    
    Returns:
        Plotly figure
    """
    if news_df.empty:
        # Create empty chart
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="white")
        )
        fig.update_layout(
            title="📊 Article Distribution by Category",
            template='plotly_dark',
            height=400,
            font=dict(color='white')
        )
        return fig
    
    # Count articles by category
    category_counts = news_df['category'].value_counts()
    
    # Define colors for categories
    colors = ['#4facfe', '#00ff88', '#ff4757', '#ffa502', '#9c88ff']
    
    fig = go.Figure(data=[go.Pie(
        labels=category_counts.index,
        values=category_counts.values,
        hole=0.4,
        marker=dict(colors=colors[:len(category_counts)]),
        textinfo='label+percent',
        textposition='outside',
        hovertemplate='<b>%{label}</b><br>' +
                     'Articles: %{value}<br>' +
                     'Percentage: %{percent}<br>' +
                     '<extra></extra>'
    )])
    
    fig.update_layout(
        title='📊 Article Distribution by Category',
        template='plotly_dark',
        height=400,
        font=dict(color='white'),
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.05
        )
    )
    
    return fig

def create_source_activity(news_df: pd.DataFrame) -> go.Figure:
    """
    Create a bar chart showing activity by news source
    
    Args:
        news_df: DataFrame with news data
    
    Returns:
        Plotly figure
    """
    if news_df.empty:
        # Create empty chart
        fig = go.Figure()
        fig.add_annotation(
            text="No source data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="white")
        )
        fig.update_layout(
            title="📡 News Source Activity",
            template='plotly_dark',
            height=400,
            font=dict(color='white')
        )
        return fig
    
    # Count articles by source
    source_counts = news_df['source'].value_counts().head(10)  # Top 10 sources
    
    fig = go.Figure(data=[go.Bar(
        x=source_counts.values,
        y=source_counts.index,
        orientation='h',
        marker=dict(
            color='#4facfe',
            line=dict(color='white', width=1)
        ),
        text=source_counts.values,
        textposition='auto',
        hovertemplate='<b>%{y}</b><br>' +
                     'Articles: %{x}<br>' +
                     '<extra></extra>'
    )])
    
    fig.update_layout(
        title='📡 News Source Activity',
        xaxis_title='Number of Articles',
        yaxis_title='News Source',
        template='plotly_dark',
        height=400,
        font=dict(color='white'),
        margin=dict(l=200)  # Make room for source names
    )
    
    return fig

def create_news_summary_metrics(news_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Create summary metrics for news data
    
    Args:
        news_df: DataFrame with news data
    
    Returns:
        Dictionary with metric values
    """
    if news_df.empty:
        return {
            'total_articles': 0,
            'avg_sentiment': 0.0,
            'sentiment_emoji': '⚪',
            'top_category': 'N/A',
            'fallback_ratio': 0.0,
            'latest_update': 'No data'
        }
    
    avg_sentiment = news_df['sentiment_score'].mean()
    
    # Determine overall sentiment
    if avg_sentiment > 0.1:
        sentiment_emoji = '🟢'
    elif avg_sentiment < -0.1:
        sentiment_emoji = '🔴'
    else:
        sentiment_emoji = '⚪'
    
    return {
        'total_articles': len(news_df),
        'avg_sentiment': avg_sentiment,
        'sentiment_emoji': sentiment_emoji,
        'top_category': news_df['category'].mode().iloc[0] if not news_df['category'].mode().empty else 'N/A',
        'fallback_ratio': news_df.get('fallback', pd.Series([False] * len(news_df))).mean(),
        'latest_update': news_df['date'].max().strftime('%Y-%m-%d %H:%M') if not news_df.empty else 'No data'
    }

def display_news_metrics(metrics: Dict[str, Any]) -> None:
    """
    Display news metrics in Streamlit columns
    
    Args:
        metrics: Dictionary with metric values
    """
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📰 Total Articles",
            value=metrics['total_articles']
        )
    
    with col2:
        st.metric(
            label=f"{metrics['sentiment_emoji']} Avg Sentiment",
            value=f"{metrics['avg_sentiment']:.3f}"
        )
    
    with col3:
        st.metric(
            label="🏷️ Top Category",
            value=metrics['top_category']
        )
    
    with col4:
        fallback_pct = metrics['fallback_ratio'] * 100
        st.metric(
            label="📡 Live Data",
            value=f"{100-fallback_pct:.0f}%"
        )

# CSS for enhanced styling
NEWS_CARD_CSS = """
<style>
.news-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 15px 40px rgba(0,0,0,0.2);
}

.news-card a:hover {
    background: linear-gradient(45deg, #764ba2, #667eea);
    transform: scale(1.05);
}

.stColumns > div {
    padding: 0 0.5rem;
}

@media (max-width: 768px) {
    .news-card {
        margin: 0.5rem 0;
        padding: 1rem;
    }
    
    .news-card h3 {
        font-size: 1rem;
        padding-right: 4rem;
    }
}
</style>
"""

def inject_news_css():
    """Inject CSS for news cards"""
    st.markdown(NEWS_CARD_CSS, unsafe_allow_html=True)
