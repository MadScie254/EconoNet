"""
EconoNet Visual Components Package
=================================

Advanced visualization components for economic data presentation.
"""

from .sentiment_radar import (
    create_sentiment_radar,
    create_provenance_footer,
    create_real_vs_synthetic_overlay,
    create_risk_alert_card,
    create_economic_heatmap
)

# News visualization components
try:
    from .news_cards import (
        create_news_grid,
        create_sentiment_timeline,
        create_category_donut,
        create_source_activity_chart
    )
except ImportError:
    # News cards not available
    pass

__all__ = [
    "create_sentiment_radar",
    "create_provenance_footer", 
    "create_real_vs_synthetic_overlay",
    "create_risk_alert_card",
    "create_economic_heatmap",
    # News components (if available)
    "create_news_grid",
    "create_sentiment_timeline", 
    "create_category_donut",
    "create_source_activity_chart",
]
