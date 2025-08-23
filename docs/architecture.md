# EconoNet Architecture

This document describes the high-level system design and architecture of EconoNet.

## System Overview

EconoNet is designed as a modular, extensible economic intelligence platform with three main layers:

```
┌─────────────────────────────────────────────────────────────┐
│                    PRESENTATION LAYER                       │
├─────────────────────────────────────────────────────────────┤
│  Streamlit Dashboards  │  Jupyter Notebooks  │  CLI Tools   │
│  - Ultra Dashboard     │  - Divine Debt Demo │  - Health    │
│  - Immersive Dashboard │  - FX Modeling      │    Check     │
│  - Enhanced App        │  - Inflation Model  │  - Pipeline  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     BUSINESS LOGIC LAYER                    │
├─────────────────────────────────────────────────────────────┤
│    EconoNet Core Package (src/econonet/)                   │
│  ┌─────────────┬─────────────┬─────────────┬─────────────┐  │
│  │   Config    │  Live APIs  │   Visual    │   Utils     │  │
│  │  Management │  Adapters   │ Components  │  Helpers    │  │
│  └─────────────┴─────────────┴─────────────┴─────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                       DATA LAYER                           │
├─────────────────────────────────────────────────────────────┤
│  External APIs          │  Cache Layer    │  Fallback Data │
│  - World Bank           │  - SQLite Cache │  - Synthetic   │
│  - ECB                  │  - TTL Control  │    Generators  │
│  - FRED                 │  - Request      │  - Quantum     │
│  - CoinGecko            │    Deduplication│    Simulation  │
│  - USGS, Wikipedia, etc │                 │                │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Configuration Management (`config.py`)

**Purpose**: Centralized configuration for operation modes, API settings, and caching.

**Key Classes**:
- `OperationMode`: Enum for offline/live/expert modes
- `APIConfig`: Per-API configuration (TTL, timeouts, etc.)
- `EconoNetConfig`: Main configuration class

**Features**:
- Environment variable integration
- Per-API configuration
- Mode-based behavior switching
- Default country/region settings

### 2. Live API Adapters (`live_apis.py`)

**Purpose**: Unified interface to all external data sources with standardized schemas.

**Key Functions**:
- `get_worldbank()` - Macroeconomic data
- `get_coingecko()` - Cryptocurrency data
- `get_usgs()` - Earthquake/risk data
- `get_wiki_views()` - Sentiment indicators
- And more...

**Features**:
- Consistent data schema across all sources
- Automatic caching with configurable TTL
- Retry logic with exponential backoff
- Graceful fallbacks to synthetic data
- Request session pooling

### 3. Visual Components (`visual/`)

**Purpose**: Reusable visualization components for dashboards and notebooks.

**Key Components**:
- `create_sentiment_radar()` - Multi-dimensional sentiment analysis
- `create_provenance_footer()` - Data source attribution
- `create_real_vs_synthetic_overlay()` - Data comparison charts
- `create_risk_alert_card()` - Risk assessment displays

### 4. Utility Functions (`utils/`)

**Purpose**: Helper functions for data processing, validation, and common operations.

## Data Flow Architecture

### 1. Request Flow

```
User Request → Dashboard → API Adapter → Cache Check → External API → Data Processing → Visualization
                    ↓         ↓            ↑              ↓              ↓
               Config Check   ↓      Cache Hit       Fallback      Schema Validation
                             ↓                      Generator           ↓
                        Mode Switch                                 Return Data
```

### 2. Cache Strategy

**Cache Hierarchy**:
1. **Memory Cache**: Fast access for frequently used data
2. **SQLite Cache**: Persistent cache across sessions
3. **Fallback Generation**: Synthetic data when APIs fail

**Cache Invalidation**:
- TTL-based expiration (configurable per API)
- Manual refresh capabilities
- Mode-based cache bypass

### 3. Error Handling

**Error Propagation**:
```
API Error → Retry Logic → Fallback Generation → UI Notification → Graceful Degradation
```

**Error Types**:
- Network timeouts
- Rate limiting
- Invalid responses
- Service unavailability

## Extensibility Design

### Adding New APIs

1. **Create adapter function** in `live_apis.py`:
```python
@api_fallback
@retry(...)
def get_new_api(params) -> pd.DataFrame:
    # Implementation
    return standardized_dataframe
```

2. **Add configuration** in `config.py`:
```python
"new_api": APIConfig(
    base_url="https://api.example.com",
    ttl_seconds=1800,
    timeout_seconds=20
)
```

3. **Update fallback generator** for synthetic data
4. **Add tests** for the new adapter

### Adding New Visualizations

1. **Create component** in `visual/`:
```python
def create_new_chart(data, title) -> go.Figure:
    # Plotly implementation
    return figure
```

2. **Add to `__init__.py`** for easy importing
3. **Document usage** and parameters

### Adding New Operation Modes

1. **Extend `OperationMode` enum**
2. **Update mode-specific logic** in relevant components
3. **Add configuration options**
4. **Update UI mode selector**

## Security Considerations

### Data Privacy
- **No personal data collection** - Only aggregate economic indicators
- **No authentication tokens** - All APIs are public/free tier
- **Local caching only** - No data transmitted to third parties

### API Security
- **Rate limiting respect** - Built-in request throttling
- **Timeout protection** - Prevents hanging requests
- **Error boundary isolation** - API failures don't crash system

### Input Validation
- **Parameter sanitization** - All user inputs validated
- **Schema validation** - Consistent data structures enforced
- **Safe fallbacks** - No arbitrary code execution in synthetic data

## Performance Considerations

### Scalability
- **Async-ready design** - Can be extended for concurrent requests
- **Modular architecture** - Components can be deployed separately
- **Caching optimization** - Reduces API load significantly

### Memory Management
- **Lazy loading** - Data fetched only when needed
- **Cache size limits** - Prevent unlimited memory growth
- **DataFrame optimization** - Efficient pandas operations

### Response Times
- **Cache-first strategy** - Sub-second response for cached data
- **Progressive loading** - UI updates as data becomes available
- **Background refresh** - Data updated in background where possible

## Deployment Architecture

### Local Development
```
Developer Machine
├── Python Environment
├── SQLite Cache
├── Streamlit Server (localhost:8501)
└── Jupyter Server (localhost:8888)
```

### Production Deployment (Future)
```
Cloud Platform
├── Container Orchestration (Docker/K8s)
├── Load Balancer
├── Application Servers
├── Shared Cache Layer (Redis)
├── Monitoring & Logging
└── CI/CD Pipeline
```

## Testing Strategy

### Unit Tests
- **API adapter tests** - Mock external services
- **Configuration tests** - Validate settings logic
- **Data validation tests** - Schema compliance
- **Fallback tests** - Synthetic data generation

### Integration Tests
- **End-to-end dashboard tests** - Full user workflows
- **API integration tests** - Real API calls (limited)
- **Cache behavior tests** - TTL and invalidation
- **Mode switching tests** - Behavior across modes

### Performance Tests
- **Load testing** - Multiple concurrent users
- **Memory profiling** - Resource usage analysis
- **API response time** - External dependency monitoring

## Monitoring & Observability

### Metrics
- **API call success rates** - Per-API reliability tracking
- **Cache hit ratios** - Caching effectiveness
- **Response times** - User experience monitoring
- **Error rates** - System health indicators

### Logging
- **Structured logging** - JSON format for analysis
- **Log levels** - DEBUG, INFO, WARNING, ERROR
- **Contextual information** - Request tracing
- **Performance metrics** - Timing information

### Alerting (Future)
- **API downtime detection**
- **Performance degradation alerts**
- **Cache miss rate monitoring**
- **User error rate tracking**

## Future Architecture Enhancements

### Microservices Evolution
- **API Gateway** - Centralized request routing
- **Service Mesh** - Inter-service communication
- **Event Streaming** - Real-time data updates
- **Distributed Caching** - Multi-region cache

### AI/ML Integration
- **Model Serving** - Prediction API endpoints
- **Feature Store** - Centralized feature management
- **Model Monitoring** - Performance tracking
- **AutoML Pipeline** - Automated model updates

### Real-time Processing
- **Stream Processing** - Live data ingestion
- **WebSocket Support** - Real-time dashboard updates
- **Event-driven Architecture** - Reactive system design
- **Change Data Capture** - Database synchronization

---

*Last Updated: August 23, 2025*  
*EconoNet Architecture Documentation v1.0*
