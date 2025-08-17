"""
NERVA: National Economic & Risk Visual Analytics
Main Entry Point
"""

# Package initialization
from .config.settings import config
from .etl.processor import CBKDataProcessor, load_cbk_data, get_data_catalog
from .models.baseline import BaselineForecaster, train_baseline_forecaster, quick_forecast

__version__ = "1.0.0"
__author__ = "NERVA Development Team"
__description__ = "National Economic & Risk Visual Analytics - CBK Decision Support System"

# Quick access functions
def run_dashboard():
    """Launch NERVA Streamlit dashboard"""
    import streamlit.web.cli as stcli
    import sys
    from pathlib import Path
    
    dashboard_path = Path(__file__).parent / "ui" / "dashboard.py"
    sys.argv = ["streamlit", "run", str(dashboard_path)]
    stcli.main()

def process_cbk_data():
    """Process all CBK data files"""
    return load_cbk_data()

def get_system_status():
    """Get NERVA system status"""
    return {
        "version": __version__,
        "config_loaded": config is not None,
        "data_path_exists": config.data.raw_data_path.exists(),
        "parquet_path_exists": config.data.parquet_path.exists()
    }

__all__ = [
    'config',
    'CBKDataProcessor', 
    'BaselineForecaster',
    'load_cbk_data',
    'get_data_catalog',
    'train_baseline_forecaster',
    'quick_forecast',
    'run_dashboard',
    'process_cbk_data',
    'get_system_status'
]
