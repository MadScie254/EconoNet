"""
EconoNet Visual Components
=========================

Advanced visualization components for sentiment analysis and data presentation.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
from typing import Dict, List, Optional, Any

def create_sentiment_radar(
    sentiment_scores: Dict[str, float],
    title: str = "Multi-Dimensional Sentiment Radar",
    show_baseline: bool = True
) -> go.Figure:
    """
    Create a radar chart for sentiment analysis
    
    Args:
        sentiment_scores: Dictionary of sentiment categories and scores (0-1)
        title: Chart title
        show_baseline: Whether to show neutral baseline
    
    Returns:
        Plotly figure object
    """
    categories = list(sentiment_scores.keys())
    values = list(sentiment_scores.values())
    
    # Close the polygon by repeating first value
    categories_closed = categories + [categories[0]]
    values_closed = values + [values[0]]
    
    fig = go.Figure()
    
    # Add main sentiment trace
    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill='toself',
        name='Current Sentiment',
        line=dict(color='#4facfe', width=3),
        fillcolor='rgba(79, 172, 254, 0.3)',
        hovertemplate='%{theta}: %{r:.2f}<extra></extra>'
    ))
    
    # Add neutral baseline if requested
    if show_baseline:
        neutral_values = [0.5] * len(categories) + [0.5]
        fig.add_trace(go.Scatterpolar(
            r=neutral_values,
            theta=categories_closed,
            mode='lines',
            name='Neutral Baseline',
            line=dict(color='#667eea', width=2, dash='dash'),
            hovertemplate='Neutral: 0.5<extra></extra>'
        ))
    
    # Update layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickvals=[0.2, 0.4, 0.6, 0.8, 1.0],
                ticktext=['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive'],
                gridcolor='rgba(255,255,255,0.2)'
            ),
            angularaxis=dict(
                gridcolor='rgba(255,255,255,0.2)'
            )
        ),
        title=dict(
            text=f'🧠 {title}',
            x=0.5,
            font=dict(size=18, color='white')
        ),
        template='plotly_dark',
        font=dict(color='white'),
        showlegend=True,
        height=500,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_provenance_footer(
    data_sources: List[Dict[str, Any]],
    last_refresh: Optional[datetime] = None
) -> str:
    """
    Create a provenance footer showing data sources and refresh time
    
    Args:
        data_sources: List of data source info dicts
        last_refresh: Last refresh timestamp
    
    Returns:
        HTML string for the footer
    """
    if last_refresh is None:
        last_refresh = datetime.now()
    
    refresh_str = last_refresh.strftime("%Y-%m-%d %H:%M UTC")
    
    # Create source badges
    source_badges = []
    for source in data_sources:
        source_name = source.get('name', 'Unknown')
        source_url = source.get('url', '#')
        is_fallback = source.get('fallback', False)
        
        badge_color = '#ff6b6b' if is_fallback else '#4facfe'
        badge_icon = '🔄' if is_fallback else '🌐'
        
        badge_html = f"""
        <a href="{source_url}" target="_blank" style="
            display: inline-block;
            background: {badge_color};
            color: white;
            padding: 2px 8px;
            border-radius: 12px;
            text-decoration: none;
            font-size: 10px;
            margin: 2px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        ">
            {badge_icon} {source_name}
        </a>
        """
        source_badges.append(badge_html)
    
    footer_html = f"""
    <div style="
        margin-top: 20px;
        padding: 10px;
        background: linear-gradient(135deg, rgba(255,255,255,0.1), rgba(255,255,255,0.05));
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.2);
        font-size: 12px;
        color: #ccc;
    ">
        <div style="margin-bottom: 5px;">
            <strong>📊 Data Sources:</strong>
        </div>
        <div style="margin-bottom: 8px;">
            {''.join(source_badges)}
        </div>
        <div style="text-align: right; font-size: 10px; opacity: 0.7;">
            🕒 Last updated: {refresh_str}
        </div>
    </div>
    """
    
    return footer_html

def create_real_vs_synthetic_overlay(
    synthetic_data: pd.Series,
    real_data: Optional[pd.DataFrame] = None,
    title: str = "Real vs Synthetic Data",
    synthetic_name: str = "Quantum Simulation",
    real_name: str = "Real World Data"
) -> go.Figure:
    """
    Create overlay visualization comparing synthetic and real data
    
    Args:
        synthetic_data: Synthetic data series
        real_data: Real data DataFrame with 'date' and value columns
        title: Chart title
        synthetic_name: Name for synthetic data series
        real_name: Name for real data series
    
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    # Add synthetic data
    fig.add_trace(go.Scatter(
        x=synthetic_data.index,
        y=synthetic_data.values,
        mode='lines',
        name=synthetic_name,
        line=dict(color='#667eea', width=2, dash='dash'),
        opacity=0.7,
        hovertemplate=f'{synthetic_name}: %{{y:.2f}}<br>Date: %{{x}}<extra></extra>'
    ))
    
    # Add real data if available
    if real_data is not None and not real_data.empty:
        value_col = real_data.columns[1] if len(real_data.columns) > 1 else real_data.columns[0]
        
        fig.add_trace(go.Scatter(
            x=real_data['date'],
            y=real_data[value_col],
            mode='lines+markers',
            name=real_name,
            line=dict(color='#00ff88', width=3),
            marker=dict(size=6, color='#00ff88'),
            hovertemplate=f'{real_name}: %{{y:.2f}}<br>Date: %{{x}}<extra></extra>'
        ))
        
        # Add event annotations for significant changes
        if len(real_data) > 1:
            values = real_data[value_col].values
            dates = real_data['date']
            
            # Find significant percentage changes
            pct_changes = np.abs(np.diff(values) / values[:-1]) * 100
            significant_threshold = np.percentile(pct_changes, 90)
            significant_changes = np.where(pct_changes > significant_threshold)[0]
            
            # Add annotations for last 3 significant events
            for idx in significant_changes[-3:]:
                change_pct = pct_changes[idx]
                fig.add_annotation(
                    x=dates.iloc[idx+1],
                    y=values[idx+1],
                    text=f"📈 {change_pct:.1f}%",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor='#ff6b6b',
                    arrowwidth=2,
                    bgcolor='rgba(255, 107, 107, 0.8)',
                    bordercolor='#ff6b6b',
                    font=dict(color='white', size=10)
                )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'🌌 {title}',
            x=0.5,
            font=dict(size=18, color='white')
        ),
        template='plotly_dark',
        font=dict(color='white'),
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            title='Date'
        ),
        yaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            title='Value'
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_risk_alert_card(
    risk_level: str,
    risk_score: float,
    risk_factors: List[str],
    title: str = "Risk Assessment"
) -> str:
    """
    Create a risk alert card component
    
    Args:
        risk_level: Risk level (LOW, MEDIUM, HIGH, CRITICAL)
        risk_score: Numeric risk score (0-1)
        risk_factors: List of contributing risk factors
        title: Card title
    
    Returns:
        HTML string for the risk card
    """
    # Define risk level colors and icons
    risk_config = {
        'LOW': {'color': '#4caf50', 'icon': '🟢', 'bg': 'rgba(76, 175, 80, 0.1)'},
        'MEDIUM': {'color': '#ff9800', 'icon': '🟡', 'bg': 'rgba(255, 152, 0, 0.1)'},
        'HIGH': {'color': '#f44336', 'icon': '🔴', 'bg': 'rgba(244, 67, 54, 0.1)'},
        'CRITICAL': {'color': '#9c27b0', 'icon': '🚨', 'bg': 'rgba(156, 39, 176, 0.1)'}
    }
    
    config = risk_config.get(risk_level, risk_config['MEDIUM'])
    
    # Create risk factors list
    factors_html = ""
    for factor in risk_factors[:5]:  # Limit to 5 factors
        factors_html += f"<li style='margin: 3px 0; font-size: 12px;'>{factor}</li>"
    
    card_html = f"""
    <div style="
        background: {config['bg']};
        border: 2px solid {config['color']};
        border-radius: 15px;
        padding: 15px;
        margin: 10px 0;
        color: white;
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
    ">
        <div style="display: flex; align-items: center; margin-bottom: 10px;">
            <div style="font-size: 24px; margin-right: 10px;">{config['icon']}</div>
            <div>
                <h3 style="margin: 0; color: {config['color']};">{title}</h3>
                <p style="margin: 2px 0; font-size: 14px; opacity: 0.8;">
                    Risk Level: <strong>{risk_level}</strong> ({risk_score:.1%})
                </p>
            </div>
        </div>
        
        {f'<div style="margin-top: 10px;"><strong>Key Factors:</strong><ul style="margin: 5px 0; padding-left: 20px;">{factors_html}</ul></div>' if risk_factors else ''}
        
        <div style="
            width: 100%;
            height: 8px;
            background: rgba(255,255,255,0.2);
            border-radius: 4px;
            margin-top: 10px;
            overflow: hidden;
        ">
            <div style="
                width: {risk_score * 100}%;
                height: 100%;
                background: {config['color']};
                transition: width 0.3s ease;
            "></div>
        </div>
    </div>
    """
    
    return card_html

def create_economic_heatmap(
    data: pd.DataFrame,
    title: str = "Economic Indicators Heatmap"
) -> go.Figure:
    """
    Create a heatmap of economic indicators
    
    Args:
        data: DataFrame with economic indicators
        title: Chart title
    
    Returns:
        Plotly figure object
    """
    # Prepare data for heatmap
    if 'date' in data.columns:
        # Pivot data for heatmap
        heatmap_data = data.pivot_table(
            index='series',
            columns='date',
            values='value',
            aggfunc='mean'
        )
        
        # Normalize data for better visualization
        normalized_data = (heatmap_data - heatmap_data.mean()) / heatmap_data.std()
        
        fig = go.Figure(data=go.Heatmap(
            z=normalized_data.values,
            x=[d.strftime('%Y-%m') for d in heatmap_data.columns],
            y=heatmap_data.index,
            colorscale='RdYlBu_r',
            hoverongaps=False,
            hovertemplate='Series: %{y}<br>Date: %{x}<br>Normalized Value: %{z:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(
                text=f'📊 {title}',
                x=0.5,
                font=dict(size=18, color='white')
            ),
            template='plotly_dark',
            font=dict(color='white'),
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig
    
    # Fallback: empty heatmap
    fig = go.Figure()
    fig.add_annotation(
        text="No data available for heatmap",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=16, color='white')
    )
    fig.update_layout(
        template='plotly_dark',
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig
