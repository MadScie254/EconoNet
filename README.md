# EconoNet: Unified Economic Intelligence Platform

[![CI](https://github.com/your-username/EconoNet/workflows/CI/badge.svg)](https://github.com/your-username/EconoNet/actions)
[![Tests](https://github.com/your-username/EconoNet/workflows/Test%20Suite/badge.svg)](https://github.com/your-username/EconoNet/actions)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

EconoNet is a comprehensive economic analysis platform that unifies live data streams, predictive modeling, and interactive visualization. Built with a focus on African economies, it provides real-time insights through no-token APIs while maintaining robust fallback capabilities.

## 🌟 Enhanced Features

### **🔄 Live Data Integration**
- **8 Unified API Adapters**: World Bank, ECB, FRED, CoinGecko, USGS, Wikipedia, Open-Meteo, IMF
- **No-Token Required**: All APIs are completely free and require no authentication
- **Smart Caching**: Configurable TTL caching with SQLite backend for optimal performance
- **Graceful Fallbacks**: Automatic synthetic data generation when APIs are unavailable

### **🎛️ Operation Modes**
- **Offline Mode**: High-quality synthetic data for development and testing
- **Live Mode**: Real-time API data with intelligent fallbacks
- **Expert Mode**: Advanced features with detailed metadata and provenance tracking

### **📊 Advanced Visualizations**
- **Sentiment Radar Charts**: Multi-dimensional economic sentiment analysis
- **Real vs Synthetic Overlays**: Clear distinction between live and simulated data
- **Risk Alert Cards**: Dynamic risk indicators with threshold monitoring
- **Provenance Footers**: Complete data lineage and source tracking

### **🚀 Interactive Dashboards**
- **Ultra Dashboard**: Quantum-themed advanced analytics interface
- **Immersive Dashboard**: Full-screen economic intelligence center  
- **Enhanced Streamlit App**: Comprehensive modeling and risk analysis

### **⚡ Core Capabilities**
- **Interactive Dashboard**: Streamlit-based interfaces for seamless navigation and analysis
- **Data Exploration**: Deep-dive tools for economic datasets with trend and correlation analysis
- **Predictive Modeling**: ARIMA, VAR, and custom time-series forecasting models
- **Financial Risk Analysis**: VaR, CVaR, Monte Carlo simulations, and stress testing
- **Notebook Integration**: Execute Jupyter notebooks with live dataframe passing

## 🏗️ Architecture

### **Core Package Structure**
```
src/econonet/
├── __init__.py           # Unified API exports
├── config.py             # Configuration management
├── live_apis.py          # 8 API adapters with fallbacks
├── utils.py              # Shared utilities
└── visual/               # Visual components
    ├── sentiment_radar.py
    ├── provenance_footer.py
    └── real_vs_synthetic.py
```

### **API System**
All API adapters return standardized DataFrames with schema:
- `date`: ISO formatted datetime
- `country`: ISO country code or name
- `series`: Data series identifier  
- `value`: Numeric value
- `source`: API source name
- `metadata`: Additional context and fallback indicators

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Git for version control  
- pip and virtualenv

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/EconoNet.git
   cd EconoNet
   ```

2. **Create and activate a virtual environment:**

   ```bash
   # For Windows
   python -m venv venv
   .\venv\Scripts\activate

   # For macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install the required packages:**

   ```bash
   pip install -r requirements.txt
   ```

### Running Applications

#### **Enhanced Dashboards**
```bash
# Ultra Dashboard (Advanced Analytics)
streamlit run ultra_dashboard_enhanced.py

# Immersive Dashboard (Full-screen Experience)  
streamlit run immersive_dashboard.py

# Enhanced Streamlit App (Comprehensive Modeling)
streamlit run enhanced_streamlit_app.py
```

#### **Jupyter Notebooks**
```bash
# Start Jupyter server
jupyter lab

# Or use the built-in notebook runner in any dashboard
```

### **🔧 Configuration**

#### **Operation Modes**
Set your preferred operation mode in the sidebar:
- **Offline**: Uses high-quality synthetic data
- **Live**: Fetches real-time data from APIs  
- **Expert**: Advanced features with detailed metadata

#### **Environment Variables**
```bash
# Optional: Set custom cache directory
export ECONET_CACHE_DIR="/path/to/cache"

# Optional: Set custom timeout (seconds)
export ECONET_API_TIMEOUT="30"
```

## 📊 API Data Sources

### **No-Token APIs** (All completely free)
- **World Bank**: GDP, inflation, trade data for 200+ countries
- **ECB**: Euro exchange rates and monetary policy indicators  
- **FRED**: US economic indicators from Federal Reserve
- **CoinGecko**: Cryptocurrency prices and market data
- **USGS**: Earthquake data and geological information
- **Wikipedia**: Page view statistics for trend analysis
- **Open-Meteo**: Weather data for economic correlation studies
- **IMF**: International monetary fund economic statistics

### **Data Standards**
All APIs return consistent DataFrame schemas:
```python
df.columns = ['date', 'country', 'series', 'value', 'source', 'metadata']
```

## 🧪 Testing & Development

### **Run Test Suite**
```bash
# Run all tests
pytest tests/ -v

# Run specific test categories  
pytest tests/test_live_apis.py -v      # API adapter tests
pytest tests/test_config.py -v         # Configuration tests
pytest tests/test_dashboards.py -v     # Dashboard integration tests
pytest tests/test_notebooks.py -v      # Notebook execution tests

# Run with coverage
pytest tests/ --cov=src/econonet --cov-report=html
```

### **Development Installation**
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install package in editable mode
pip install -e .
```

### **CI/CD Pipeline**
The repository includes GitHub Actions workflows:
- **Lint & Test**: Code quality and unit tests
- **Notebook Smoke Tests**: Validate notebook execution
- **Streamlit Import Tests**: Verify dashboard functionality

## 📁 Project Structure

```
EconoNet/
├── src/econonet/              # Core EconoNet package
│   ├── __init__.py            # Unified API exports
│   ├── config.py              # Configuration management  
│   ├── live_apis.py           # 8 API adapters with caching
│   ├── utils.py               # Shared utilities
│   └── visual/                # Advanced visualization components
├── dashboard/                 # Dashboard applications
│   ├── ultra_dashboard_enhanced.py    # Quantum-themed analytics
│   ├── immersive_dashboard.py         # Full-screen experience
│   └── enhanced_streamlit_app.py      # Comprehensive modeling
├── notebooks/                 # Jupyter analysis notebooks
│   ├── EDA.ipynb             # Exploratory data analysis
│   ├── FX_modeling.ipynb     # Foreign exchange modeling
│   ├── Inflation_modeling.ipynb # Inflation forecasting
│   └── Liquidity_modeling.ipynb # Liquidity analysis
├── data/                      # Data storage
│   ├── raw/                   # Original CSV datasets
│   ├── processed/             # Cleaned data
│   └── cleaned/               # Final analysis-ready data
├── tests/                     # Comprehensive test suite
│   ├── test_live_apis.py      # API adapter testing
│   ├── test_config.py         # Configuration testing
│   ├── test_dashboards.py     # Dashboard integration tests
│   └── test_notebooks.py      # Notebook execution tests
├── docs/                      # Documentation
│   ├── data_sources.md        # API documentation
│   ├── architecture.md        # System architecture
│   └── usage_examples.md      # Usage examples
├── .github/workflows/         # CI/CD pipelines
│   ├── lint-test.yml          # Code quality and testing
│   ├── notebook-smoke-test.yml # Notebook validation
│   └── streamlit-import-test.yml # Dashboard functionality
├── requirements.txt           # Production dependencies
├── requirements-dev.txt       # Development dependencies
├── pyproject.toml            # Project configuration
└── README.md                 # This file
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### **Development Workflow**
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and add tests
4. Run the test suite: `pytest tests/ -v`
5. Submit a pull request

### **Code Standards**
- Follow PEP 8 style guidelines
- Add type hints for new functions
- Include docstrings for public APIs
- Maintain test coverage above 85%

## 📖 Documentation

- **[Data Sources](docs/data_sources.md)**: Complete API documentation
- **[Architecture](docs/architecture.md)**: System design and patterns  
- **[Usage Examples](docs/usage_examples.md)**: Code examples and tutorials

## 🔗 Links

- **Repository**: [GitHub](https://github.com/your-username/EconoNet)
- **Issues**: [Bug Reports & Feature Requests](https://github.com/your-username/EconoNet/issues)
- **Discussions**: [Community Forum](https://github.com/your-username/EconoNet/discussions)

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Data Providers**: World Bank, ECB, FRED, CoinGecko, USGS, Wikipedia, Open-Meteo, IMF
- **Open Source Libraries**: Streamlit, Plotly, Pandas, Scikit-learn, Requests
- **African Economic Data**: Focus on Kenya, Nigeria, South Africa, Ghana, Tanzania, Uganda

---

**EconoNet** - Unified Economic Intelligence for Africa 🌍
   source venv/bin/activate
   ```

3. **Install the required packages:**

   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

Launch the Streamlit application by running the following command from the root directory:

```bash
streamlit run dashboard/app.py
```

The application will open in your default web browser.

## 📂 Project Structure

| Name             | Description                                                 |
| :--------------- | :---------------------------------------------------------- |
| `dashboard/`     | Contains the Streamlit application scripts.                 |
| `├── app.py`     | The main entry point for the Streamlit app.                 |
| `└── pages/`     | Additional pages for the dashboard (e.g., Notebook Explorer).|
| `data/`          | Directory for storing raw, processed, and cleaned data.     |
| `models/`        | Saved machine learning models.                              |
| `notebooks/`     | Jupyter notebooks for detailed analysis and experimentation.|
| `src/`           | Source code for data processing, modeling, and utilities.   |
| `└── utils/`     | Utility scripts, including the notebook runner.             |
| `requirements.txt`| A list of Python packages required for the project.         |
| `README.md`      | This file.                                                  |


## 🔬 Analyses and Notebooks

The `notebooks/` directory contains detailed walkthroughs of the core analyses performed in this project:

1. **Data Exploration (`Data_Exploration.ipynb`)**:
   - Covers EDA, feature engineering, correlation analysis, and data preprocessing.
2. **Predictive Modeling (`Predictive_Models.ipynb`)**:
   - Demonstrates time-series forecasting using models like ARIMA.
3. **Risk Analysis (`Risk_Analysis.ipynb`)**:
   - Provides a framework for financial risk assessment using VaR, CVaR, and Monte Carlo methods.

These notebooks can be run interactively within the "Notebook Explorer" section of the Streamlit application.

