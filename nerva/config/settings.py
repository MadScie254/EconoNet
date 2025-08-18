"""
NERVA Configuration Management
GODMODE_X: Execution-first configuration system
"""

from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv

load_dotenv()

class DataConfig(BaseModel):
    """Data pipeline configuration"""
    raw_data_path: Path = Field(default=Path("data/raw"))
    processed_data_path: Path = Field(default=Path("data/processed"))
    cleaned_data_path: Path = Field(default=Path("data/cleaned"))
    parquet_path: Path = Field(default=Path("data/parquet"))
    
    # CBK Data Sources
    cbk_data_files: Dict[str, str] = {
        "gdp": "Annual GDP.csv",
        "cbr": "Central Bank Rate (CBR)  .csv", 
        "interbank": "Interbank Rates  Volumes.csv",
        "fx_monthly": "Monthly exchange rate (end period).csv",
        "fx_average": "Monthly Exchange rate (period average).csv",
        "remittances": "Diaspora Remittances.csv",
        "mobile_payments": "Mobile Payments.csv",
        "public_debt": "Public Debt.csv",
        "trade_summary": "Foreign Trade Summary (Ksh Million).csv",
        "treasury_bills": "Issues of Treasury Bills.csv",
        "treasury_bonds": "Issues of Treasury Bonds.csv",
        "repo_rates": "Repo and Reverse Repo .csv",
        "commercial_rates": "Commercial Banks Weighted Average Rates ().csv",
        "transactions": "Value of Transactions (Kshs. Millions).csv",
        "revenue_expenditure": "Revenue and Expenditure.csv"
    }
    
    # Data quality thresholds
    min_completeness: float = 0.7
    max_outlier_zscore: float = 4.0

class ModelConfig(BaseModel):
    """ML Model configuration"""
    # Forecasting
    forecast_horizons: list = [1, 3, 6, 12]  # months
    train_test_split: float = 0.8
    cv_folds: int = 5
    
    # Transformer config
    hidden_dim: int = 256
    num_heads: int = 8
    num_layers: int = 6
    dropout: float = 0.1
    
    # Graph network config
    graph_embedding_dim: int = 128
    graph_hidden_dim: int = 256
    
    # Training
    batch_size: int = 32
    learning_rate: float = 1e-4
    max_epochs: int = 100
    patience: int = 10

class UIConfig(BaseModel):
    """Streamlit UI configuration"""
    page_title: str = "NERVA: National Economic & Risk Visual Analytics"
    page_icon: str = "N"
    layout: str = "wide"
    
    # Dashboard refresh rates
    realtime_refresh_seconds: int = 30
    heavy_model_cache_hours: int = 6
    
    # Color schemes
    risk_colors: Dict[str, str] = {
        "low": "#2ecc71",
        "medium": "#f39c12", 
        "high": "#e74c3c",
        "critical": "#8e44ad"
    }

class SystemConfig(BaseModel):
    """System-wide configuration"""
    # Paths
    project_root: Path = Field(default=Path(__file__).parent.parent.parent)
    log_level: str = "INFO"
    
    # Performance
    n_jobs: int = -1
    gpu_enabled: bool = True
    
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    # Redis/Celery
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    celery_broker: str = os.getenv("CELERY_BROKER", "redis://localhost:6379/0")

class NERVAConfig:
    """Main NERVA configuration manager"""
    
    def __init__(self):
        self.data = DataConfig()
        self.model = ModelConfig()
        self.ui = UIConfig()
        self.system = SystemConfig()
        
        # Ensure directories exist
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories"""
        for path in [
            self.data.processed_data_path,
            self.data.cleaned_data_path,
            self.data.parquet_path,
            Path("logs"),
            Path("artifacts"),
            Path("models_registry")
        ]:
            path.mkdir(parents=True, exist_ok=True)
    
    def get_data_file_path(self, file_key: str) -> Path:
        """Get full path to a data file"""
        if file_key not in self.data.cbk_data_files:
            raise ValueError(f"Unknown data file key: {file_key}")
        
        return self.data.raw_data_path / self.data.cbk_data_files[file_key]
    
    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary"""
        return {
            "data": self.data.dict(),
            "model": self.model.dict(),
            "ui": self.ui.dict(),
            "system": self.system.dict()
        }

# Global configuration instance
config = NERVAConfig()
