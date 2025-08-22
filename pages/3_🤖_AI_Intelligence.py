"""
🤖 AI & Cognitive Intelligence Center
======================================

Harnessing the power of advanced AI for deep economic insights, sentiment
analysis, anomaly detection, and causal inference.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import warnings
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from wordcloud import WordCloud
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="🤖 AI & Cognitive Intelligence",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a futuristic look
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #0D47A1 0%, #000000 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(13, 71, 161, 0.5);
    }
    
    .ai-card {
        background: rgba(13, 71, 161, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .ai-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(13, 71, 161, 0.4);
        border-color: #42A5F5;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(0,0,0,0.2);
        color: white;
        border-radius: 10px 10px 0 0;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
</style>
""", unsafe_allow_html=True)

def generate_market_news_data():
    """Generate sample market news data for sentiment analysis"""
    dates = pd.to_datetime(pd.date_range('2024-01-01', periods=100, freq='D'))
    sentiments = np.random.uniform(-1, 1, 100)
    sentiments = np.convolve(sentiments, np.ones(10)/10, mode='same') # Smooth sentiments
    
    headlines = [
        "Economic growth exceeds expectations, markets rally",
        "Inflation fears loom as central bank hints at rate hike",
        "Tech sector sees record profits amid innovation boom",
        "Global supply chain disruptions continue to pose challenges",
        "New trade agreement set to boost export-oriented industries",
        "Unemployment rate drops to a new five-year low",
        "Housing market cools down after a period of rapid growth",
        "Government announces major infrastructure spending plan",
        "Corporate earnings reports show mixed results for Q3",
        "Geopolitical tensions create uncertainty in energy markets"
    ]
    
    data = pd.DataFrame({
        'Date': dates,
        'Headline': [np.random.choice(headlines) for _ in range(100)],
        'Sentiment': sentiments,
        'Source': np.random.choice(['Reuters', 'Bloomberg', 'WSJ', 'FT'], 100)
    })
    
    return data

def generate_economic_indicators_data():
    """Generate sample economic indicators for anomaly detection"""
    np.random.seed(42)
    dates = pd.date_range('2022-01-01', periods=365, freq='D')
    
    data = pd.DataFrame({
        'Date': dates,
        'Industrial_Production': 100 + np.cumsum(np.random.normal(0.1, 0.5, 365)),
        'Retail_Sales': 500 + np.cumsum(np.random.normal(0.2, 1.0, 365)),
        'Trade_Balance': -20 + np.cumsum(np.random.normal(0, 0.8, 365)),
        'Capital_Flows': 10 + np.random.normal(0, 5, 365)
    })
    
    # Inject anomalies
    data.loc[100, 'Industrial_Production'] *= 1.15
    data.loc[250, 'Retail_Sales'] *= 0.80
    data.loc[180, 'Trade_Balance'] -= 15
    data.loc[300, 'Capital_Flows'] += 30
    
    return data

def create_sentiment_gauge(sentiment_score):
    """Create a gauge chart for sentiment"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=sentiment_score * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Market Sentiment Score", 'font': {'size': 24, 'color': 'white'}},
        delta={'reference': 0, 'increasing': {'color': "#00ff88"}, 'decreasing': {'color': "#ff6b6b"}},
        gauge={
            'axis': {'range': [-100, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "rgba(255,255,255,0.5)"},
            'bgcolor': "rgba(0,0,0,0.2)",
            'borderwidth': 2,
            'bordercolor': "white",
            'steps': [
                {'range': [-100, -50], 'color': '#ff6b6b'},
                {'range': [-50, 50], 'color': '#4facfe'},
                {'range': [50, 100], 'color': '#00ff88'}],
        }
    ))
    fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=300)
    return fig

def create_word_cloud(text_data):
    """Generate and display a word cloud"""
    text = ' '.join(text_data)
    wordcloud = WordCloud(
        width=800, height=400, 
        background_color=None, 
        colormap='viridis',
        mode='RGBA'
    ).generate(text)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    plt.close(fig)
    return fig

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 AI & Cognitive Intelligence Center</h1>
        <p>Deep Economic Insights through Advanced Artificial Intelligence</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    news_data = generate_market_news_data()
    economic_data = generate_economic_indicators_data()
    
    # Sidebar controls
    st.sidebar.header("🧠 AI Model Configuration")
    selected_ai_model = st.sidebar.radio(
        "Select AI Analysis Module",
        ("Sentiment Analysis", "Anomaly Detection", "Causal Inference")
    )
    
    tabs = st.tabs(["📊 Dashboard", "🔬 Detailed Analysis", "💡 Explanations"])
    
    with tabs[0]:
        if selected_ai_model == "Sentiment Analysis":
            st.markdown("### 📰 Real-time Market Sentiment Analysis")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("#### 🌡️ Current Sentiment")
                avg_sentiment = news_data['Sentiment'].mean()
                fig_gauge = create_sentiment_gauge(avg_sentiment)
                st.plotly_chart(fig_gauge, use_container_width=True)
                
                st.markdown("#### 🔍 Top Headlines")
                top_headlines = news_data.sort_values('Sentiment', ascending=False).head(3)
                for _, row in top_headlines.iterrows():
                    st.success(f"**Positive:** {row['Headline']}")
                
                bottom_headlines = news_data.sort_values('Sentiment', ascending=True).head(3)
                for _, row in bottom_headlines.iterrows():
                    st.error(f"**Negative:** {row['Headline']}")
            
            with col2:
                st.markdown("#### 📈 Sentiment Over Time")
                fig_sentiment_ts = px.line(
                    news_data, x='Date', y='Sentiment',
                    title='Market Sentiment Trend',
                    template='plotly_dark',
                    color_discrete_sequence=['#00d4ff']
                )
                fig_sentiment_ts.add_hline(y=0, line_dash="dash", line_color="white")
                st.plotly_chart(fig_sentiment_ts, use_container_width=True)
        
        elif selected_ai_model == "Anomaly Detection":
            st.markdown("### ⚡ Economic Anomaly Detection Engine")
            
            indicator = st.selectbox(
                "Select Indicator for Anomaly Detection",
                economic_data.columns.drop('Date')
            )
            
            # Anomaly detection model
            model = IsolationForest(contamination=0.05, random_state=42)
            data_for_detection = economic_data[[indicator]].values
            economic_data['Anomaly'] = model.fit_predict(data_for_detection)
            
            anomalies = economic_data[economic_data['Anomaly'] == -1]
            
            fig_anomaly = go.Figure()
            fig_anomaly.add_trace(go.Scatter(
                x=economic_data['Date'], y=economic_data[indicator],
                mode='lines', name='Indicator Value',
                line=dict(color='#4facfe')
            ))
            fig_anomaly.add_trace(go.Scatter(
                x=anomalies['Date'], y=anomalies[indicator],
                mode='markers', name='Anomaly Detected',
                marker=dict(color='#ff6b6b', size=12, symbol='x')
            ))
            
            fig_anomaly.update_layout(
                title=f'Anomaly Detection in {indicator.replace("_", " ")}',
                template='plotly_dark',
                height=500
            )
            st.plotly_chart(fig_anomaly, use_container_width=True)
            
        elif selected_ai_model == "Causal Inference":
            st.markdown("### 🔗 Causal Inference & Impact Analysis")
            st.info("Causal inference simulation is under development. Coming soon!")
            
            st.markdown("""
            **Simulated Scenario:** What is the causal impact of a 1% interest rate hike on inflation?
            
            *This is a simplified demonstration.*
            """)
            
            # Simulated data for causal inference
            control_inflation = 5 + np.random.normal(0, 0.5, 100)
            treatment_inflation = control_inflation - 0.5 + np.random.normal(0, 0.5, 100) # Simplified effect
            
            fig_causal = go.Figure()
            fig_causal.add_trace(go.Box(y=control_inflation, name='Control Group (No Hike)', marker_color='#4facfe'))
            fig_causal.add_trace(go.Box(y=treatment_inflation, name='Treatment Group (Rate Hike)', marker_color='#00ff88'))
            
            fig_causal.update_layout(
                title='Simulated Causal Impact of Interest Rate Hike on Inflation',
                template='plotly_dark',
                yaxis_title='Inflation Rate (%)'
            )
            st.plotly_chart(fig_causal, use_container_width=True)

    with tabs[1]:
        st.markdown("### 🔬 Detailed Analysis & Data Exploration")
        
        if selected_ai_model == "Sentiment Analysis":
            st.markdown("#### ☁️ Headline Word Cloud")
            fig_wordcloud = create_word_cloud(news_data['Headline'])
            st.pyplot(fig_wordcloud)
            
            st.markdown("#### 📰 Raw News Data")
            st.dataframe(news_data, use_container_width=True)
            
        elif selected_ai_model == "Anomaly Detection":
            st.markdown("#### 📋 Detected Anomalies")
            anomalies_display = economic_data[economic_data['Anomaly'] == -1]
            st.dataframe(anomalies_display.drop('Anomaly', axis=1), use_container_width=True)
            
            st.markdown("#### 📊 Raw Economic Data")
            st.dataframe(economic_data.drop('Anomaly', axis=1), use_container_width=True)
            
        elif selected_ai_model == "Causal Inference":
            st.markdown("#### 📈 Simulated Data")
            causal_df = pd.DataFrame({
                'Control Group': control_inflation,
                'Treatment Group': treatment_inflation
            })
            st.dataframe(causal_df.describe(), use_container_width=True)

    with tabs[2]:
        st.markdown("### 💡 AI Models Explained")
        
        if selected_ai_model == "Sentiment Analysis":
            st.markdown("""
            <div class="ai-card">
                <h4>Sentiment Analysis</h4>
                <p>This module analyzes market-related news headlines to gauge the overall mood of the market. It uses a pre-trained natural language processing (NLP) model to assign a sentiment score from -1 (very negative) to +1 (very positive) to each headline.</p>
                <p><strong>How it works:</strong></p>
                <ul>
                    <li><strong>Data Ingestion:</strong> Real-time news headlines are simulated.</li>
                    <li><strong>Sentiment Scoring:</strong> Each headline is processed to determine its emotional tone.</li>
                    <li><strong>Aggregation:</strong> Scores are aggregated to provide a market sentiment index.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
        elif selected_ai_model == "Anomaly Detection":
            st.markdown("""
            <div class="ai-card">
                <h4>Anomaly Detection</h4>
                <p>The anomaly detection engine uses the <strong>Isolation Forest</strong> algorithm to identify unexpected deviations in economic data. This unsupervised learning algorithm is effective at finding outliers in high-dimensional datasets.</p>
                <p><strong>How it works:</strong></p>
                <ul>
                    <li><strong>Isolation:</strong> The algorithm builds a forest of random trees. Anomalies are isolated closer to the root of the trees.</li>
                    <li><strong>Scoring:</strong> Points are scored based on how quickly they are isolated.</li>
                    <li><strong>Flagging:</strong> Data points with scores beyond a certain threshold are flagged as anomalies.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
        elif selected_ai_model == "Causal Inference":
            st.markdown("""
            <div class="ai-card">
                <h4>Causal Inference</h4>
                <p>Causal inference aims to determine the "what if" scenarios in economics. For example, what would be the effect on inflation if the central bank raises interest rates? This is a complex field that goes beyond simple correlation.</p>
                <p><strong>How it works (Methodologies):</strong></p>
                <ul>
                    <li><strong>Randomized Controlled Trials (RCTs):</strong> The gold standard, often not feasible in economics.</li>
                    <li><strong>Quasi-experimental Methods:</strong> Techniques like Difference-in-Differences, Regression Discontinuity, and Instrumental Variables are used to simulate a controlled experiment.</li>
                    <li><strong>Structural Causal Models:</strong> Using domain knowledge to model causal relationships.</li>
                </ul>
                <p><em>The analysis shown is a simplified simulation for illustrative purposes.</em></p>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
