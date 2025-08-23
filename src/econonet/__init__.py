"""
EconoNet - Advanced Economic Intelligence Platform
================================================

A comprehensive economic analysis platform with real-time API integration,
quantum-inspired modeling, and immersive visualizations.

Key Features:
- Live API data integration (World Bank, ECB, FRED, CoinGecko, etc.)
- Three operation modes: offline, live, expert
- Unified data schemas and caching
- Advanced sentiment analysis and risk monitoring
- Notebook integration with live data feeds
"""

__version__ = "1.0.0"
__author__ = "EconoNet Team"

# Core imports
from .config import (
    OperationMode,
    EconoNetConfig,
    get_config,
    set_mode,
    is_live_mode,
    is_offline_mode,
    is_expert_mode
)

from .live_apis import (
    get_worldbank,
    get_ecb,
    get_fred_csv,
    get_coingecko,
    get_usgs,
    get_wiki_views,
    get_open_meteo,
    get_imf,
    get_all_data,
    validate_data_schema
)

# Visual components (will be created)
try:
    from .visual import (
        create_sentiment_radar,
        create_provenance_footer,
        create_real_vs_synthetic_overlay
    )
except ImportError:
    # Visual components not yet available
    pass

__all__ = [
    # Version info
    "__version__",
    "__author__",
    
    # Configuration
    "OperationMode",
    "EconoNetConfig", 
    "get_config",
    "set_mode",
    "is_live_mode",
    "is_offline_mode",
    "is_expert_mode",
    
    # API functions
    "get_worldbank",
    "get_ecb", 
    "get_fred_csv",
    "get_coingecko",
    "get_usgs",
    "get_wiki_views",
    "get_open_meteo",
    "get_imf",
    "get_all_data",
    "validate_data_schema",
]
