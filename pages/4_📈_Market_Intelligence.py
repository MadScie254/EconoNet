"""
Market Intelligence Page - Real-time Economic Market Analysis
============================================================

Advanced market intelligence platform providing real-time analysis,
sentiment tracking, policy impact assessment, and market trend prediction.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import requests
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Market Intelligence - EconoNet",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import custom modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Define default colors for compatibility
ECONET_COLORS = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', '#00f2fe']

class MarketIntelligenceEngine:
    """
    Advanced Market Intelligence Engine
    
    Provides real-time market analysis including:
    - Economic sentiment tracking
    - Policy impact assessment
    - Market trend prediction
    - Cross-market correlation analysis
    - Real-time data integration
    """
    
    def __init__(self):
        self.market_data = {}
        self.sentiment_scores = {}
        self.policy_impacts = {}
        self.market_correlations = {}
        
    def get_market_sentiment(self, keywords: List[str]) -> Dict[str, Any]:
        """
        Analyze market sentiment from economic indicators and news
        
        Args:
            keywords: List of economic keywords to track
            
        Returns:
            Dict containing sentiment scores and trends
        """
        # Simulated sentiment analysis (in production, integrate with news APIs)
        sentiment_data = {}
        
        for keyword in keywords:
            # Generate realistic sentiment scores
            base_sentiment = np.random.normal(0, 0.3)  # Neutral bias
            trend = np.random.choice([-1, 0, 1], p=[0.3, 0.4, 0.3])  # Trend direction
            
            sentiment_data[keyword] = {
                'current_sentiment': base_sentiment,
                'trend': trend,
                'confidence': np.random.uniform(0.6, 0.95),
                'volume': np.random.randint(100, 1000),
                'classification': self._classify_sentiment(base_sentiment)
            }
        
        return sentiment_data
    
    def analyze_policy_impact(self, policy_type: str, 
                            economic_indicators: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze potential policy impacts on economic indicators
        
        Args:
            policy_type: Type of policy (monetary, fiscal, trade)
            economic_indicators: Historical economic data
            
        Returns:
            Policy impact analysis
        """
        impact_analysis = {
            'policy_type': policy_type,
            'expected_impacts': {},
            'confidence_scores': {},
            'timeline': {},
            'risk_assessment': {}
        }
        
        # Policy impact matrices (simplified simulation)
        policy_impacts = {
            'monetary': {
                'Interest_Rate': 0.8,
                'Inflation': -0.6,
                'Exchange_Rate': 0.4,
                'GDP': 0.3,
                'Investment': 0.7
            },
            'fiscal': {
                'GDP': 0.7,
                'Employment': 0.6,
                'Debt': 0.9,
                'Inflation': 0.3,
                'Investment': 0.5
            },
            'trade': {
                'Exchange_Rate': 0.8,
                'Trade_Balance': 0.9,
                'GDP': 0.4,
                'Inflation': 0.2,
                'Employment': 0.3
            }
        }
        
        if policy_type in policy_impacts:
            for indicator, impact_strength in policy_impacts[policy_type].items():
                impact_analysis['expected_impacts'][indicator] = {
                    'impact_magnitude': impact_strength,
                    'direction': np.random.choice(['positive', 'negative'], 
                                                p=[0.6, 0.4]),
                    'timeline_months': np.random.randint(3, 18),
                    'confidence': impact_strength * np.random.uniform(0.8, 1.0)
                }
        
        return impact_analysis
    
    def calculate_market_correlations(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate cross-market correlations and relationships"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return {'error': 'Insufficient numeric columns for correlation analysis'}
        
        # Calculate correlation matrix
        corr_matrix = data[numeric_cols].corr()
        
        # Find strongest correlations
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > 0.3:  # Significant correlation threshold
                    corr_pairs.append({
                        'pair': (corr_matrix.columns[i], corr_matrix.columns[j]),
                        'correlation': corr_value,
                        'strength': self._classify_correlation_strength(abs(corr_value))
                    })
        
        # Sort by correlation strength
        corr_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        return {
            'correlation_matrix': corr_matrix,
            'top_correlations': corr_pairs[:10],
            'market_relationships': self._analyze_market_relationships(corr_pairs)
        }
    
    def generate_market_signals(self, data: pd.DataFrame, 
                               lookback_period: int = 30) -> Dict[str, Any]:
        """
        Generate trading/investment signals based on market analysis
        
        Args:
            data: Market/economic data
            lookback_period: Number of periods to analyze
            
        Returns:
            Market signals and recommendations
        """
        signals = {}
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            series = data[col].dropna()
            if len(series) < lookback_period:
                continue
            
            recent_data = series.tail(lookback_period)
            
            # Technical analysis signals
            signals[col] = {
                'trend_signal': self._calculate_trend_signal(recent_data),
                'momentum_signal': self._calculate_momentum_signal(recent_data),
                'volatility_signal': self._calculate_volatility_signal(recent_data),
                'mean_reversion_signal': self._calculate_mean_reversion_signal(recent_data),
                'overall_signal': None,
                'confidence': 0.0
            }
            
            # Combine signals for overall recommendation
            signal_scores = [
                signals[col]['trend_signal']['score'],
                signals[col]['momentum_signal']['score'],
                signals[col]['volatility_signal']['score'],
                signals[col]['mean_reversion_signal']['score']
            ]
            
            overall_score = np.mean(signal_scores)
            signals[col]['overall_signal'] = self._classify_signal(overall_score)
            signals[col]['confidence'] = 1 - np.std(signal_scores) / 2  # Higher std = lower confidence
        
        return signals
    
    def _classify_sentiment(self, sentiment_score: float) -> str:
        """Classify sentiment score into categories"""
        if sentiment_score > 0.3:
            return "Very Positive"
        elif sentiment_score > 0.1:
            return "Positive"
        elif sentiment_score > -0.1:
            return "Neutral"
        elif sentiment_score > -0.3:
            return "Negative"
        else:
            return "Very Negative"
    
    def _classify_correlation_strength(self, correlation: float) -> str:
        """Classify correlation strength"""
        if correlation > 0.8:
            return "Very Strong"
        elif correlation > 0.6:
            return "Strong"
        elif correlation > 0.4:
            return "Moderate"
        elif correlation > 0.2:
            return "Weak"
        else:
            return "Very Weak"
    
    def _analyze_market_relationships(self, corr_pairs: List[Dict]) -> Dict[str, Any]:
        """Analyze market relationships from correlation data"""
        relationships = {
            'strong_positive': [],
            'strong_negative': [],
            'market_clusters': [],
            'risk_factors': []
        }
        
        for pair in corr_pairs:
            corr_value = pair['correlation']
            pair_name = f"{pair['pair'][0]} - {pair['pair'][1]}"
            
            if corr_value > 0.7:
                relationships['strong_positive'].append({
                    'pair': pair_name,
                    'correlation': corr_value,
                    'implication': "Strong co-movement, diversification risk"
                })
            elif corr_value < -0.7:
                relationships['strong_negative'].append({
                    'pair': pair_name,
                    'correlation': corr_value,
                    'implication': "Strong negative relationship, hedge potential"
                })
        
        return relationships
    
    def _calculate_trend_signal(self, series: pd.Series) -> Dict[str, Any]:
        """Calculate trend-based signal"""
        short_ma = series.rolling(5).mean().iloc[-1]
        long_ma = series.rolling(20).mean().iloc[-1]
        current_price = series.iloc[-1]
        
        if current_price > short_ma > long_ma:
            signal = "Strong Buy"
            score = 1.0
        elif current_price > short_ma:
            signal = "Buy"
            score = 0.5
        elif current_price < short_ma < long_ma:
            signal = "Strong Sell"
            score = -1.0
        elif current_price < short_ma:
            signal = "Sell"
            score = -0.5
        else:
            signal = "Hold"
            score = 0.0
        
        return {
            'signal': signal,
            'score': score,
            'short_ma': short_ma,
            'long_ma': long_ma
        }
    
    def _calculate_momentum_signal(self, series: pd.Series) -> Dict[str, Any]:
        """Calculate momentum-based signal"""
        momentum = series.pct_change(5).iloc[-1]  # 5-period momentum
        
        if momentum > 0.05:
            signal = "Strong Positive Momentum"
            score = 1.0
        elif momentum > 0.02:
            signal = "Positive Momentum"
            score = 0.5
        elif momentum < -0.05:
            signal = "Strong Negative Momentum"
            score = -1.0
        elif momentum < -0.02:
            signal = "Negative Momentum"
            score = -0.5
        else:
            signal = "Neutral Momentum"
            score = 0.0
        
        return {
            'signal': signal,
            'score': score,
            'momentum': momentum
        }
    
    def _calculate_volatility_signal(self, series: pd.Series) -> Dict[str, Any]:
        """Calculate volatility-based signal"""
        volatility = series.pct_change().rolling(10).std().iloc[-1]
        avg_volatility = series.pct_change().std()
        
        vol_ratio = volatility / avg_volatility if avg_volatility > 0 else 1
        
        if vol_ratio < 0.5:
            signal = "Low Volatility"
            score = 0.3  # Slightly positive for stability
        elif vol_ratio < 1.5:
            signal = "Normal Volatility"
            score = 0.0
        elif vol_ratio < 2.0:
            signal = "High Volatility"
            score = -0.3
        else:
            signal = "Very High Volatility"
            score = -0.7
        
        return {
            'signal': signal,
            'score': score,
            'volatility': volatility,
            'vol_ratio': vol_ratio
        }
    
    def _calculate_mean_reversion_signal(self, series: pd.Series) -> Dict[str, Any]:
        """Calculate mean reversion signal"""
        mean_price = series.mean()
        current_price = series.iloc[-1]
        std_dev = series.std()
        
        z_score = (current_price - mean_price) / std_dev if std_dev > 0 else 0
        
        if z_score > 2:
            signal = "Extremely Overbought"
            score = -0.8  # Sell signal
        elif z_score > 1:
            signal = "Overbought"
            score = -0.4
        elif z_score < -2:
            signal = "Extremely Oversold"
            score = 0.8  # Buy signal
        elif z_score < -1:
            signal = "Oversold"
            score = 0.4
        else:
            signal = "Fair Value"
            score = 0.0
        
        return {
            'signal': signal,
            'score': score,
            'z_score': z_score
        }
    
    def _classify_signal(self, score: float) -> str:
        """Classify overall signal score"""
        if score > 0.5:
            return "Strong Buy"
        elif score > 0.2:
            return "Buy"
        elif score > -0.2:
            return "Hold"
        elif score > -0.5:
            return "Sell"
        else:
            return "Strong Sell"

def create_market_dashboard(intelligence_data: Dict[str, Any]) -> None:
    """Create comprehensive market intelligence dashboard"""
    
    # Market overview metrics
    st.subheader("📊 Market Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Simulated market index
        market_index = np.random.normal(2500, 50)
        daily_change = np.random.normal(0.5, 1.5)
        
        st.metric(
            "Market Index",
            f"{market_index:.0f}",
            delta=f"{daily_change:+.1f}%"
        )
    
    with col2:
        # Volatility index
        volatility = np.random.uniform(15, 25)
        vol_change = np.random.normal(0, 2)
        
        st.metric(
            "Volatility Index",
            f"{volatility:.1f}",
            delta=f"{vol_change:+.1f}"
        )
    
    with col3:
        # Economic sentiment
        sentiment_score = np.random.uniform(45, 75)
        sentiment_change = np.random.normal(0, 3)
        
        st.metric(
            "Economic Sentiment",
            f"{sentiment_score:.0f}",
            delta=f"{sentiment_change:+.1f}"
        )
    
    with col4:
        # Policy uncertainty
        uncertainty = np.random.uniform(20, 80)
        uncertainty_change = np.random.normal(0, 5)
        
        st.metric(
            "Policy Uncertainty",
            f"{uncertainty:.0f}",
            delta=f"{uncertainty_change:+.1f}"
        )

def create_sentiment_analysis(sentiment_data: Dict[str, Any]) -> None:
    """Create sentiment analysis visualization"""
    
    st.subheader("📰 Economic Sentiment Analysis")
    
    if not sentiment_data:
        st.info("Generate sentiment analysis to see results")
        return
    
    # Create sentiment chart
    keywords = list(sentiment_data.keys())
    sentiments = [data['current_sentiment'] for data in sentiment_data.values()]
    confidences = [data['confidence'] for data in sentiment_data.values()]
    
    fig = go.Figure()
    
    # Sentiment bars
    colors = [ECONET_COLORS['positive'] if s > 0 else ECONET_COLORS['negative'] 
              for s in sentiments]
    
    fig.add_trace(go.Bar(
        x=keywords,
        y=sentiments,
        marker_color=colors,
        text=[f"{s:.2f}" for s in sentiments],
        textposition='auto',
        name='Sentiment Score'
    ))
    
    # Add confidence as secondary axis
    fig.add_trace(go.Scatter(
        x=keywords,
        y=[c * max(abs(min(sentiments)), max(sentiments)) for c in confidences],
        mode='markers',
        marker=dict(
            size=10,
            color=ECONET_COLORS['accent'],
            line=dict(width=2, color='white')
        ),
        name='Confidence Level',
        yaxis='y2'
    ))
    
    fig.update_layout(
        title="Economic Sentiment by Topic",
        xaxis_title="Economic Topics",
        yaxis_title="Sentiment Score",
        yaxis2=dict(
            title="Confidence Level",
            overlaying='y',
            side='right',
            range=[0, 1]
        ),
        template="plotly_dark",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Sentiment details
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📈 Most Positive")
        positive_topics = sorted(sentiment_data.items(), 
                               key=lambda x: x[1]['current_sentiment'], 
                               reverse=True)[:3]
        
        for topic, data in positive_topics:
            classification = data['classification']
            confidence = data['confidence']
            st.success(f"**{topic}**: {classification} (Confidence: {confidence:.1%})")
    
    with col2:
        st.markdown("#### 📉 Most Negative")
        negative_topics = sorted(sentiment_data.items(), 
                               key=lambda x: x[1]['current_sentiment'])[:3]
        
        for topic, data in negative_topics:
            classification = data['classification']
            confidence = data['confidence']
            st.error(f"**{topic}**: {classification} (Confidence: {confidence:.1%})")

def create_policy_impact_analysis(policy_data: Dict[str, Any]) -> None:
    """Create policy impact analysis visualization"""
    
    st.subheader("🏛️ Policy Impact Analysis")
    
    if not policy_data or 'expected_impacts' not in policy_data:
        st.info("Run policy impact analysis to see results")
        return
    
    impacts = policy_data['expected_impacts']
    
    # Create impact visualization
    indicators = list(impacts.keys())
    magnitudes = [impact['impact_magnitude'] for impact in impacts.values()]
    timelines = [impact['timeline_months'] for impact in impacts.values()]
    directions = [impact['direction'] for impact in impacts.values()]
    
    # Impact magnitude chart
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure()
        
        colors = [ECONET_COLORS['positive'] if direction == 'positive' 
                 else ECONET_COLORS['negative'] for direction in directions]
        
        fig.add_trace(go.Bar(
            x=indicators,
            y=magnitudes,
            marker_color=colors,
            text=[f"{m:.1f}" for m in magnitudes],
            textposition='auto'
        ))
        
        fig.update_layout(
            title=f"Expected {policy_data['policy_type'].title()} Policy Impact",
            xaxis_title="Economic Indicators",
            yaxis_title="Impact Magnitude",
            template="plotly_dark"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Timeline analysis
        fig = px.scatter(
            x=timelines,
            y=magnitudes,
            color=directions,
            size=[abs(m) for m in magnitudes],
            hover_name=indicators,
            title="Impact Timeline vs Magnitude",
            labels={'x': 'Timeline (months)', 'y': 'Impact Magnitude'},
            color_discrete_map={
                'positive': ECONET_COLORS['positive'],
                'negative': ECONET_COLORS['negative']
            }
        )
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
    
    # Policy recommendations
    st.markdown("#### 💡 Policy Recommendations")
    
    high_impact_indicators = [
        (indicator, data) for indicator, data in impacts.items()
        if data['impact_magnitude'] > 0.6
    ]
    
    if high_impact_indicators:
        for indicator, data in high_impact_indicators:
            direction_emoji = "📈" if data['direction'] == 'positive' else "📉"
            st.info(
                f"{direction_emoji} **{indicator}**: Expected {data['direction']} impact "
                f"of {data['impact_magnitude']:.1f} magnitude over {data['timeline_months']} months"
            )
    else:
        st.info("No high-impact indicators identified for this policy type")

def create_market_signals_display(signals_data: Dict[str, Any]) -> None:
    """Display market signals and recommendations"""
    
    st.subheader("🎯 Market Signals & Recommendations")
    
    if not signals_data:
        st.info("Generate market signals to see recommendations")
        return
    
    # Create signals summary
    signal_summary = []
    for indicator, signals in signals_data.items():
        signal_summary.append({
            'Indicator': indicator,
            'Overall Signal': signals['overall_signal'],
            'Confidence': signals['confidence'],
            'Trend': signals['trend_signal']['signal'],
            'Momentum': signals['momentum_signal']['signal']
        })
    
    signals_df = pd.DataFrame(signal_summary)
    
    # Color code signals
    def color_signal(val):
        if 'Buy' in val:
            return f'background-color: {ECONET_COLORS["positive"]}; color: white'
        elif 'Sell' in val:
            return f'background-color: {ECONET_COLORS["negative"]}; color: white'
        else:
            return f'background-color: {ECONET_COLORS["neutral"]}; color: white'
    
    styled_df = signals_df.style.applymap(
        color_signal, 
        subset=['Overall Signal', 'Trend', 'Momentum']
    ).format({'Confidence': '{:.1%}'})
    
    st.dataframe(styled_df, use_container_width=True)
    
    # Signal strength visualization
    col1, col2 = st.columns(2)
    
    with col1:
        # Overall signal distribution
        signal_counts = signals_df['Overall Signal'].value_counts()
        
        fig = px.pie(
            values=signal_counts.values,
            names=signal_counts.index,
            title="Signal Distribution",
            color_discrete_map={
                'Strong Buy': ECONET_COLORS['positive'],
                'Buy': ECONET_COLORS['positive'],
                'Hold': ECONET_COLORS['neutral'],
                'Sell': ECONET_COLORS['negative'],
                'Strong Sell': ECONET_COLORS['negative']
            }
        )
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Confidence levels
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=signals_df['Indicator'],
            y=signals_df['Confidence'],
            marker_color=ECONET_COLORS['accent'],
            text=[f"{c:.1%}" for c in signals_df['Confidence']],
            textposition='auto'
        ))
        
        fig.update_layout(
            title="Signal Confidence Levels",
            xaxis_title="Indicators",
            yaxis_title="Confidence",
            template="plotly_dark"
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Main app
def main():
    """Main application function"""
    
    st.title("📈 Market Intelligence")
    st.markdown("Real-time market analysis, sentiment tracking, and policy impact assessment.")

    # File uploader for market analysis
    st.sidebar.header("Upload Data for Analysis")
    uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file:
        # data_cleaner = DataCleaner()  # Commented out for now
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # cleaned_df = data_cleaner.clean_dataframe(df)  # Commented out for now
            cleaned_df = df  # Use original dataframe for now
            st.sidebar.success("File uploaded successfully!")
            st.session_state['market_data'] = cleaned_df
        except Exception as e:
            st.sidebar.error(f"Error processing file: {e}")

    # Initialize engine
    engine = MarketIntelligenceEngine()
    
    # Sidebar controls
    st.sidebar.header("🎛️ Market Intelligence Settings")
    
    # Analysis type selection
    analysis_type = st.sidebar.selectbox(
        "Analysis Type",
        ["Market Overview", "Sentiment Analysis", "Policy Impact", "Market Signals", "Correlations"],
        help="Choose the type of market analysis to perform"
    )
    
    # Data source
    if 'data' not in st.session_state:
        st.warning("Please upload data in the main Dashboard first.")
        st.stop()
    
    data = st.session_state.data
    
    # Market intelligence configuration
    st.sidebar.subheader("📊 Analysis Configuration")
    
    # Economic keywords for sentiment analysis
    economic_keywords = st.sidebar.multiselect(
        "Economic Topics",
        ["GDP", "Inflation", "Employment", "Trade", "Monetary Policy", 
         "Fiscal Policy", "Exchange Rate", "Interest Rates", "Investment"],
        default=["GDP", "Inflation", "Employment"],
        help="Select topics for sentiment analysis"
    )
    
    # Policy type for impact analysis
    policy_type = st.sidebar.selectbox(
        "Policy Type",
        ["monetary", "fiscal", "trade"],
        help="Select policy type for impact analysis"
    )
    
    # Analysis period
    lookback_period = st.sidebar.slider(
        "Analysis Period (days)",
        min_value=7,
        max_value=90,
        value=30,
        help="Number of days to analyze"
    )
    
    # Real-time updates
    auto_refresh = st.sidebar.checkbox(
        "Auto Refresh",
        value=False,
        help="Automatically refresh market data"
    )
    
    if auto_refresh:
        refresh_interval = st.sidebar.slider(
            "Refresh Interval (seconds)",
            min_value=30,
            max_value=300,
            value=60
        )
    
    # Main content based on analysis type
    if analysis_type == "Market Overview":
        # Create market dashboard
        intelligence_data = {}  # Placeholder for real market data
        create_market_dashboard(intelligence_data)
        
        # Market trends
        st.markdown("---")
        st.subheader("📈 Market Trends")
        
        if len(data.select_dtypes(include=[np.number]).columns) > 0:
            # Create trend analysis
            numeric_cols = data.select_dtypes(include=[np.number]).columns[:4]  # Limit to 4
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=numeric_cols,
                vertical_spacing=0.1
            )
            
            for i, col in enumerate(numeric_cols):
                row = (i // 2) + 1
                col_num = (i % 2) + 1
                
                # Add trend line
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data[col],
                        mode='lines',
                        name=col,
                        line=dict(color=ECONET_COLORS['primary'])
                    ),
                    row=row, col=col_num
                )
                
                # Add moving average
                ma = data[col].rolling(12).mean()
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=ma,
                        mode='lines',
                        name=f'{col} MA',
                        line=dict(color=ECONET_COLORS['accent'], dash='dash')
                    ),
                    row=row, col=col_num
                )
            
            fig.update_layout(
                title="Economic Indicators Trends",
                template="plotly_dark",
                height=600,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    elif analysis_type == "Sentiment Analysis":
        st.markdown("### 📰 Economic Sentiment Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col2:
            if st.button("🔄 Generate Sentiment Analysis", type="primary"):
                with st.spinner("Analyzing economic sentiment..."):
                    sentiment_data = engine.get_market_sentiment(economic_keywords)
                    st.session_state.sentiment_data = sentiment_data
                    st.success("✅ Sentiment analysis completed!")
        
        if 'sentiment_data' in st.session_state:
            create_sentiment_analysis(st.session_state.sentiment_data)
        else:
            st.info("Click 'Generate Sentiment Analysis' to analyze economic sentiment")
    
    elif analysis_type == "Policy Impact":
        st.markdown("### 🏛️ Policy Impact Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col2:
            if st.button("🎯 Analyze Policy Impact", type="primary"):
                with st.spinner("Analyzing policy impacts..."):
                    policy_data = engine.analyze_policy_impact(policy_type, data)
                    st.session_state.policy_data = policy_data
                    st.success("✅ Policy analysis completed!")
        
        if 'policy_data' in st.session_state:
            create_policy_impact_analysis(st.session_state.policy_data)
        else:
            st.info("Click 'Analyze Policy Impact' to assess policy effects")
    
    elif analysis_type == "Market Signals":
        st.markdown("### 🎯 Market Signals Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col2:
            if st.button("📊 Generate Market Signals", type="primary"):
                with st.spinner("Generating market signals..."):
                    signals_data = engine.generate_market_signals(data, lookback_period)
                    st.session_state.signals_data = signals_data
                    st.success("✅ Market signals generated!")
        
        if 'signals_data' in st.session_state:
            create_market_signals_display(st.session_state.signals_data)
        else:
            st.info("Click 'Generate Market Signals' to see trading recommendations")
    
    elif analysis_type == "Correlations":
        st.markdown("### 🔗 Cross-Market Correlation Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col2:
            if st.button("🔍 Analyze Correlations", type="primary"):
                with st.spinner("Analyzing market correlations..."):
                    corr_data = engine.calculate_market_correlations(data)
                    st.session_state.corr_data = corr_data
                    st.success("✅ Correlation analysis completed!")
        
        if 'corr_data' in st.session_state:
            corr_data = st.session_state.corr_data
            
            if 'error' not in corr_data:
                # Correlation heatmap
                fig = px.imshow(
                    corr_data['correlation_matrix'],
                    title="Economic Indicators Correlation Matrix",
                    color_continuous_scale="RdBu",
                    aspect="auto",
                    text_auto=True
                )
                fig.update_layout(template="plotly_dark", height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Top correlations
                st.subheader("🔗 Strongest Relationships")
                
                top_corr = corr_data['top_correlations'][:5]
                for i, corr in enumerate(top_corr, 1):
                    pair_name = f"{corr['pair'][0]} - {corr['pair'][1]}"
                    corr_value = corr['correlation']
                    strength = corr['strength']
                    
                    color = "green" if corr_value > 0 else "red"
                    st.markdown(
                        f"**{i}.** {pair_name}: **{corr_value:.3f}** ({strength})"
                    )
            else:
                st.error(corr_data['error'])
        else:
            st.info("Click 'Analyze Correlations' to see market relationships")
    
    # Auto-refresh functionality
    if auto_refresh and analysis_type == "Market Overview":
        time.sleep(refresh_interval)
        st.experimental_rerun()

if __name__ == "__main__":
    main()
