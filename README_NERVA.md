# NERVA: National Economic & Risk Visual Analytics

## 🧠 Executive Summary

NERVA transforms the Central Bank of Kenya's data archive into a real-time "National Financial Nervous System" - providing forecasts, systemic risk scores, anomaly detection, and prescriptive policy recommendations through a single unified Streamlit command center.

## 🚀 Sprint 1 Deliverables (COMPLETE)

### ✅ Core Infrastructure
- **Configuration System**: Centralized settings management with data paths, model parameters, and UI configuration
- **ETL Engine**: Automated processing of 34 CBK datasets with quality assessment and parquet storage
- **Baseline Models**: Multi-model ensemble (LightGBM, Random Forest, ARIMA, ETS) for economic forecasting
- **Streamlit UI**: Interactive dashboard with KPI cards, data catalog, and forecasting interface

### ✅ Data Processing Capabilities
- **Automated Schema Detection**: Smart column type inference and standardization
- **Data Quality Assessment**: Completeness scoring, outlier detection, and recommendations
- **Multi-format Support**: CSV, Excel with robust encoding detection
- **Feature Engineering**: Lagged variables, rolling statistics, technical indicators

### ✅ Forecasting System
- **Multi-horizon Predictions**: 1, 3, 6, 12-month forecasts with ensemble averaging
- **Performance Metrics**: RMSE, MAE, MAPE with model comparison
- **Interactive Training**: Real-time model training through UI
- **Prediction Visualization**: Multi-model forecast charts

## 🏗️ System Architecture

```
EconoNet/
├── nerva/                          # NERVA Core System
│   ├── config/settings.py          # Configuration management
│   ├── etl/processor.py            # Data ingestion & processing
│   ├── models/baseline.py          # Baseline forecasting models
│   ├── ui/dashboard.py             # Streamlit command center
│   └── __init__.py                 # Package initialization
├── requirements_nerva.txt          # Dependencies
└── launch_nerva.py                 # Quick launcher
```

## 🎯 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements_nerva.txt
```

### 2. Launch NERVA Dashboard
```bash
python launch_nerva.py
```
**OR with automatic dependency installation:**
```bash
python launch_nerva.py --install-deps
```

### 3. Access Dashboard
Open browser to: `http://localhost:8501`

## 📊 Current Capabilities

### Data Catalog
- **34 CBK Datasets** automatically processed
- **Quality Scoring** (0-1 scale) with recommendations
- **Schema Detection** for dates, numerics, currencies
- **Parquet Storage** for optimized performance

### Baseline Forecasting
- **LightGBM**: Gradient boosting with feature importance
- **Random Forest**: Ensemble tree-based prediction
- **Linear Regression**: Simple statistical baseline
- **ARIMA**: Time series autoregressive modeling
- **ETS**: Exponential smoothing
- **Ensemble**: Averaged predictions across all models

### Dashboard Features
- **KPI Cards**: Systemic Risk Index, Inflation/FX forecasts, Reserve adequacy
- **Data Overview**: Interactive catalog with quality color-coding
- **Model Training**: One-click ensemble training
- **Multi-horizon Forecasts**: 1-24 month predictions
- **Performance Metrics**: Model comparison and validation

## 🔮 Sprint 2 Roadmap (Next Phase)

### Graph Networks & Systemic Risk
- **Temporal Graph Transformer**: Bank-sector-macro dynamic network modeling
- **Contagion Detection**: Risk propagation pathways and scoring
- **Network Visualization**: Interactive pyvis/sigma.js integration
- **Anomaly Detection**: Graph autoencoder for pattern discovery

### NLP & Document Intelligence
- **PDF Processing**: CBK reports and policy document extraction
- **Auto-summarization**: 5-sentence executive briefs
- **Entity Recognition**: Policy changes, dates, numerical facts
- **Sentiment Analysis**: Market communication impact

## 🔧 Sprint 3 Roadmap (Final Phase)

### Advanced Analytics
- **Probabilistic Forecasting**: Uncertainty quantification with prediction intervals
- **Causal Inference**: Policy counterfactual simulation
- **Prescriptive RL**: Constrained reinforcement learning for policy recommendations
- **Real-time APIs**: FastAPI backend with Celery job processing

### Production Features
- **Model Registry**: MLflow integration for version control
- **Automated Retraining**: Data drift detection and pipeline triggers
- **Export Capabilities**: PDF policy briefs and data downloads
- **Performance Monitoring**: Prediction accuracy tracking

## 📈 Evaluation Metrics

### Sprint 1 Acceptance Criteria ✅
- [x] ETL processes all 34 CBK datasets with >70% quality scores
- [x] Baseline ensemble achieves reasonable RMSE on holdout data
- [x] Streamlit dashboard loads in <4 seconds for cached data
- [x] Interactive forecasting with multi-model comparison
- [x] Data catalog with automated quality assessment

### Performance Benchmarks
- **Data Processing**: ~34 files in <30 seconds
- **Model Training**: <60 seconds for baseline ensemble
- **Dashboard Response**: <2 seconds for cached operations
- **Memory Usage**: <2GB for full dataset processing

## 🔍 Key CBK Datasets Integrated

1. **Monetary Policy**: CBR rates, repo operations, discount window
2. **Exchange Rates**: Monthly FX rates (end period & average)
3. **Banking**: Commercial bank rates, interbank activity
4. **Payments**: Mobile payments, cheques, EFTs, transactions
5. **Trade**: Export/import values, trade summaries
6. **Debt**: Public debt, treasury bills/bonds issuance
7. **Macroeconomic**: GDP, remittances, revenue/expenditure

## 🛠️ Technology Stack

- **Backend**: Python 3.8+, pandas, numpy, scikit-learn
- **ML/AI**: LightGBM, PyTorch, transformers, statsmodels
- **Frontend**: Streamlit, Plotly, interactive visualizations
- **Data**: Parquet storage, automated ETL pipelines
- **Future**: FastAPI, Redis, Celery, Docker deployment

## 🎪 Demo Script (10 minutes)

1. **Launch** (1 min): `python launch_nerva.py`
2. **Data Overview** (2 min): Explore catalog, quality scores
3. **Select Dataset** (1 min): Choose FX rates or inflation data
4. **Train Models** (3 min): One-click ensemble training
5. **View Predictions** (2 min): Multi-horizon forecasts
6. **Interpret Results** (1 min): Model performance comparison

---

**NERVA v1.0** - Built for Central Bank of Kenya Decision Support

*Sprint 1 Complete ✅ | Sprint 2 Ready 🚀*
