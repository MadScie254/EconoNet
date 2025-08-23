#!/usr/bin/env python3
"""
Simple test script for the EconoNet News Module
"""

import sys
from pathlib import Path

# Add src to path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir / 'src'))

def test_news_module():
    try:
        import importlib.util
        news_file = current_dir / 'src' / 'econonet' / 'live_news.py'
        if news_file.exists():
            print('✅ News module file found')
            spec = importlib.util.spec_from_file_location('live_news', news_file)
            live_news = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(live_news)
            
            # Test sentiment analysis
            score, label, emoji = live_news.analyze_sentiment('Great news for fintech!')
            print(f'Sentiment test: {score:.2f} | {label} | {emoji}')
            
            # Test fallback data
            df = live_news.generate_fallback_news()
            print(f'Fallback data: {len(df)} articles generated')
            
            # Test stats
            stats = live_news.get_news_summary_stats(df)
            print(f'Stats: {stats["total_articles"]} articles, avg sentiment: {stats["avg_sentiment"]:.2f}')
            
            print('✅ All basic tests passed!')
            return True
        else:
            print('❌ News module file not found at:', news_file)
            return False
    except Exception as e:
        print(f'❌ Error: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_news_module()
