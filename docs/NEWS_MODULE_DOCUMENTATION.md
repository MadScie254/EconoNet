# EconoNet News & Insights Module

## 📰 Overview

The EconoNet News & Insights module provides real-time fintech news aggregation with advanced sentiment analysis and interactive visualization capabilities. This module integrates seamlessly with the main EconoNet dashboard and provides comprehensive news intelligence for economic analysis.

## 🌟 Features

### News Data Sources
- **GNews API**: Global news aggregation with search capabilities
- **Yahoo Finance RSS**: Financial market news and updates
- **CryptoPanic**: Cryptocurrency and blockchain news
- **Reuters Business RSS**: Professional business news feed
- **BBC Business RSS**: International business and economic news
- **Fallback System**: Synthetic news generation when APIs are unavailable

### Sentiment Analysis
- **TextBlob Integration**: Natural language processing for sentiment scoring
- **Emoji Indicators**: Visual sentiment representation (🟢 Bullish, 🔴 Bearish, ⚪ Neutral)
- **Sentiment Timeline**: Track sentiment trends over time
- **Radar Charts**: Multi-dimensional sentiment analysis

### Interactive Dashboard
- **News Grid**: Clean, card-based news layout with sentiment badges
- **Filter System**: Filter by category, region, sentiment, and keywords
- **Analytics Tabs**: Comprehensive charts and statistics
- **Video Integration**: Embedded fintech education videos
- **Source Testing**: Real-time API health monitoring

## 📁 File Structure

```
src/econonet/
├── live_news.py              # Core news data module (368 lines)
├── visual/
│   └── news_cards.py         # Visual components (584 lines)
└── pages/
    └── fintech_news.py       # Dashboard page (445 lines)

tests/
├── test_live_news.py         # Comprehensive test suite
└── test_news_basic.py        # Basic functionality tests
```

## 🔧 Installation & Setup

### Dependencies
```bash
pip install requests feedparser textblob pandas plotly streamlit
python -m textblob.corpora.download  # Download sentiment data
```

### Environment Setup
No API keys required! The module uses token-free endpoints:
- GNews public API (with fallback to RSS)
- Yahoo Finance RSS feeds
- Public RSS feeds from major news sources

### Integration with Main Dashboard
The module integrates automatically with `ultra_dashboard_enhanced.py`:
```python
# Added to main dashboard
tab10 = "📰 News & Insights"

# Import and run
from econonet.pages.fintech_news import run_fintech_news_page
run_fintech_news_page()
```

## 🚀 Usage Examples

### Basic News Fetching
```python
from econonet.live_news import get_fintech_news

# Get latest fintech news
df = get_fintech_news(
    query="fintech",
    region="global",
    categories=['fintech', 'crypto'],
    limit=20
)
print(f"Retrieved {len(df)} articles")
```

### Sentiment Analysis
```python
from econonet.live_news import analyze_sentiment

score, label, emoji = analyze_sentiment("Great news for fintech innovation!")
print(f"Sentiment: {label} {emoji} (Score: {score:.2f})")
# Output: Sentiment: bullish 🟢 (Score: 0.85)
```

### Search Functionality
```python
from econonet.live_news import search_news

# Search for specific topics
results = search_news(
    query="blockchain",
    region="africa",
    sentiment="bullish"
)
```

### Summary Statistics
```python
from econonet.live_news import get_news_summary_stats

stats = get_news_summary_stats(news_df)
print(f"Total articles: {stats['total_articles']}")
print(f"Average sentiment: {stats['avg_sentiment']:.2f}")
```

## 📊 Dashboard Features

### Main Interface
- **Sidebar Filters**: Category, region, sentiment, keyword filtering
- **News Grid**: Card-based layout with sentiment indicators
- **Real-time Metrics**: Article count, sentiment score, active sources

### Analytics Tabs
1. **Overview**: Key metrics and recent headlines
2. **Sentiment Timeline**: Time-series sentiment analysis
3. **Source Analysis**: Source activity and reliability
4. **Category Distribution**: News topic breakdown
5. **Educational Videos**: Embedded fintech learning content

### Interactive Charts
- **Sentiment Timeline**: Plotly line chart with sentiment trends
- **Sentiment Radar**: Multi-category sentiment visualization
- **Category Donut**: News distribution by category
- **Source Activity**: Bar chart of source contribution

## 🛡️ Fallback System

### Graceful Degradation
When APIs are unavailable, the system provides:
- Synthetic news data with realistic patterns
- Fallback flag indicators for data quality
- User notifications about offline mode
- Consistent schema across all data sources

### Data Quality Indicators
```python
# Each news DataFrame includes:
df['fallback']       # Boolean: True if synthetic data
df['last_refresh']   # Timestamp of data generation
df['confidence']     # Data confidence score (0-1)
```

## 🔍 API Endpoints

### Token-Free Sources
1. **GNews Public API**
   - Endpoint: `https://gnews.io/api/v4/search`
   - Fallback: Google News RSS
   - Rate Limit: Handled with exponential backoff

2. **Yahoo Finance RSS**
   - Endpoint: `https://feeds.finance.yahoo.com/rss/2.0/headline`
   - Categories: Business, technology, markets

3. **CryptoPanic RSS**
   - Endpoint: `https://cryptopanic.com/api/v1/posts/`
   - Focus: Cryptocurrency and blockchain news

4. **Reuters Business RSS**
   - Endpoint: `https://www.reuters.com/business/feed/`
   - Quality: Professional journalism

5. **BBC Business RSS**
   - Endpoint: `https://feeds.bbci.co.uk/news/business/rss.xml`
   - Coverage: International business news

## 🧪 Testing

### Test Coverage
- **Unit Tests**: Individual function testing
- **Integration Tests**: Multi-source aggregation
- **Performance Tests**: Sentiment analysis throughput
- **Error Handling**: Network failure scenarios
- **Data Quality**: Schema validation and fallback testing

### Running Tests
```bash
# Basic functionality tests
python tests/test_news_basic.py

# Comprehensive test suite (requires responses library)
pytest tests/test_live_news.py -v

# Performance benchmarks
pytest tests/test_live_news.py::TestNewsPerformance -v
```

## 📈 Performance Metrics

### Benchmarks
- **Sentiment Analysis**: >10 texts/second
- **News Fetching**: <5 seconds for 50 articles
- **Dashboard Loading**: <3 seconds typical
- **Memory Usage**: <50MB for 1000 articles

### Optimization Features
- **Caching**: 1-hour TTL for API responses
- **Pagination**: Lazy loading for large datasets
- **Async Support**: Background data refreshing
- **Rate Limiting**: Respectful API usage

## 🔧 Configuration

### NewsConfig Class
```python
class NewsConfig:
    timeout_seconds = 10
    max_retries = 3
    cache_ttl_minutes = 60
    max_articles_per_source = 50
    enable_fallback = True
    sentiment_threshold = 0.1
```

### Customization Options
- Modify source priorities in `live_news.py`
- Adjust sentiment thresholds for classification
- Configure cache duration for performance
- Enable/disable specific news sources

## 🚨 Error Handling

### Common Issues & Solutions

1. **Import Errors**
   ```python
   # Check if module is available
   try:
       from econonet.live_news import get_fintech_news
   except ImportError:
       print("News module not found")
   ```

2. **API Rate Limits**
   - Automatic exponential backoff
   - Fallback to alternative sources
   - User notification of rate limit status

3. **Network Connectivity**
   - Graceful fallback to synthetic data
   - Offline mode indicators
   - Cached data utilization

4. **Data Quality Issues**
   - Schema validation for all sources
   - Data cleaning and normalization
   - Confidence scoring for reliability

## 🎯 Integration Points

### Main Dashboard Integration
- Added as tab 10: "📰 News & Insights"
- Consistent styling with EconoNet theme
- Responsive design for all screen sizes

### Data Pipeline Integration
- News sentiment feeds into economic models
- RSS data enriches market analysis
- Real-time updates for trading indicators

### Notebook Integration
- Export news data to Jupyter notebooks
- API functions available in notebook environment
- Visualization helpers for custom analysis

## 🔮 Future Enhancements

### Planned Features
1. **Advanced NLP**: Named Entity Recognition (NER)
2. **Topic Modeling**: LDA/BERT-based topic extraction
3. **Predictive Analytics**: News-based market prediction
4. **Social Media**: Twitter/Reddit sentiment integration
5. **Multilingual Support**: Non-English news sources
6. **Export Features**: PDF reports and data downloads

### Roadmap
- **Phase 1**: Core functionality (✅ Complete)
- **Phase 2**: Advanced analytics (🔄 In Progress)
- **Phase 3**: Social media integration (📅 Planned)
- **Phase 4**: Predictive modeling (📅 Future)

## 📞 Support & Contributing

### Getting Help
- Check test files for usage examples
- Review error messages for specific issues
- Ensure all dependencies are installed
- Verify network connectivity for APIs

### Contributing
- Follow existing code style and patterns
- Add tests for new functionality
- Update documentation for changes
- Test fallback mechanisms thoroughly

## 📜 License & Attribution

This module is part of the EconoNet project and follows the same licensing terms. News content is aggregated from public sources with proper attribution and respect for terms of service.

### Data Sources Attribution
- GNews: Global news aggregation service
- Yahoo Finance: Financial market data
- Reuters: Professional journalism
- BBC: International news coverage
- CryptoPanic: Cryptocurrency news community

---

**EconoNet News & Insights Module** - Transforming how economic intelligence meets real-time information analysis. 🚀📰
