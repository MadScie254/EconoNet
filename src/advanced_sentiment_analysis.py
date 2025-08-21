"""
Advanced Real-Time Economic Sentiment Analysis System
====================================================

Ultra-sophisticated natural language processing and sentiment analysis
for economic news, social media, and market sentiment prediction
"""

import pandas as pd
import numpy as np
import re
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class AdvancedSentimentAnalyzer:
    """
    Advanced sentiment analysis with economic domain expertise
    """
    
    def __init__(self):
        # Economic sentiment lexicon
        self.economic_lexicon = {
            # Positive economic terms
            'growth': 0.8, 'expansion': 0.7, 'boom': 0.9, 'recovery': 0.6,
            'bullish': 0.8, 'optimistic': 0.7, 'surge': 0.8, 'rally': 0.7,
            'strong': 0.6, 'robust': 0.7, 'healthy': 0.6, 'rising': 0.5,
            'gains': 0.6, 'profit': 0.7, 'increase': 0.5, 'improve': 0.6,
            'positive': 0.6, 'upward': 0.5, 'higher': 0.5, 'strengthen': 0.6,
            'accelerate': 0.6, 'advance': 0.5, 'climb': 0.5, 'soar': 0.8,
            
            # Negative economic terms
            'recession': -0.9, 'depression': -1.0, 'crash': -0.9, 'collapse': -0.9,
            'bearish': -0.8, 'pessimistic': -0.7, 'decline': -0.6, 'fall': -0.5,
            'weak': -0.6, 'fragile': -0.7, 'struggling': -0.7, 'falling': -0.5,
            'losses': -0.6, 'deficit': -0.6, 'decrease': -0.5, 'worsen': -0.6,
            'negative': -0.6, 'downward': -0.5, 'lower': -0.5, 'weaken': -0.6,
            'decelerate': -0.6, 'retreat': -0.5, 'plunge': -0.8, 'tumble': -0.7,
            'crisis': -0.8, 'turmoil': -0.7, 'uncertainty': -0.5, 'volatile': -0.4,
            'inflation': -0.3, 'unemployment': -0.6, 'debt': -0.4, 'risk': -0.3,
            
            # Neutral but important economic terms
            'stable': 0.1, 'steady': 0.1, 'maintain': 0.0, 'neutral': 0.0,
            'flat': 0.0, 'unchanged': 0.0, 'sideways': 0.0, 'consolidate': 0.0
        }
        
        # Economic entities and their importance weights
        self.economic_entities = {
            'central bank': 1.0, 'federal reserve': 1.0, 'fed': 1.0, 'ecb': 1.0,
            'bank of england': 1.0, 'boj': 1.0, 'pboc': 1.0, 'rbi': 1.0,
            'gdp': 0.9, 'inflation': 0.9, 'unemployment': 0.9, 'cpi': 0.8,
            'ppi': 0.7, 'pmi': 0.7, 'retail sales': 0.7, 'housing starts': 0.6,
            'consumer confidence': 0.7, 'business confidence': 0.7,
            'trade deficit': 0.6, 'current account': 0.6, 'fiscal deficit': 0.7,
            'government debt': 0.6, 'corporate earnings': 0.8, 'oil prices': 0.7,
            'stock market': 0.8, 'bond market': 0.7, 'forex': 0.7, 'currency': 0.6
        }
        
        # Sentiment modifiers
        self.intensifiers = {
            'very': 1.3, 'extremely': 1.5, 'highly': 1.3, 'significantly': 1.4,
            'substantially': 1.4, 'dramatically': 1.5, 'sharply': 1.4,
            'moderately': 0.8, 'slightly': 0.6, 'somewhat': 0.7, 'relatively': 0.8
        }
        
        self.negations = ['not', 'no', 'never', 'none', 'neither', 'nor', 'nothing']
        
    def preprocess_text(self, text):
        """
        Advanced text preprocessing for economic content
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s\.\!\?\-\%\$]', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Handle contractions
        contractions = {
            "won't": "will not", "can't": "cannot", "n't": " not",
            "'re": " are", "'ve": " have", "'ll": " will", "'d": " would"
        }
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        return text.strip()
    
    def extract_economic_entities(self, text):
        """
        Extract economic entities and their context
        """
        entities = []
        text_lower = text.lower()
        
        for entity, weight in self.economic_entities.items():
            if entity in text_lower:
                # Find context around the entity
                pattern = rf'.{{0,50}}{re.escape(entity)}.{{0,50}}'
                matches = re.finditer(pattern, text_lower)
                
                for match in matches:
                    context = match.group(0)
                    entities.append({
                        'entity': entity,
                        'weight': weight,
                        'context': context,
                        'position': match.start()
                    })
        
        return entities
    
    def calculate_sentiment_score(self, text):
        """
        Calculate comprehensive sentiment score
        """
        text = self.preprocess_text(text)
        words = text.split()
        
        total_score = 0
        word_count = 0
        entity_boost = 0
        
        # Extract economic entities
        entities = self.extract_economic_entities(text)
        entity_weights = sum([e['weight'] for e in entities])
        
        i = 0
        while i < len(words):
            word = words[i]
            
            # Check for sentiment word
            if word in self.economic_lexicon:
                score = self.economic_lexicon[word]
                
                # Check for intensifiers before the word
                if i > 0 and words[i-1] in self.intensifiers:
                    score *= self.intensifiers[words[i-1]]
                
                # Check for negations
                negation_found = False
                for j in range(max(0, i-3), i):  # Look back up to 3 words
                    if words[j] in self.negations:
                        negation_found = True
                        break
                
                if negation_found:
                    score *= -0.5  # Flip and dampen
                
                total_score += score
                word_count += 1
            
            i += 1
        
        # Normalize by word count
        if word_count > 0:
            base_sentiment = total_score / word_count
        else:
            base_sentiment = 0
        
        # Apply entity boost
        if entity_weights > 0:
            entity_boost = min(entity_weights * 0.1, 0.3)  # Cap at 0.3
        
        final_sentiment = base_sentiment + entity_boost
        
        # Clip to [-1, 1] range
        final_sentiment = max(-1, min(1, final_sentiment))
        
        return {
            'sentiment_score': final_sentiment,
            'base_sentiment': base_sentiment,
            'entity_boost': entity_boost,
            'sentiment_words_count': word_count,
            'entities_found': len(entities),
            'entity_details': entities
        }
    
    def analyze_multiple_texts(self, texts):
        """
        Analyze multiple texts and return aggregated sentiment
        """
        results = []
        
        for text in texts:
            if isinstance(text, str) and text.strip():
                result = self.calculate_sentiment_score(text)
                result['text_length'] = len(text)
                result['original_text'] = text[:100] + "..." if len(text) > 100 else text
                results.append(result)
        
        if not results:
            return {'error': 'No valid texts to analyze'}
        
        # Aggregate results
        total_sentiment = sum([r['sentiment_score'] for r in results])
        avg_sentiment = total_sentiment / len(results)
        
        # Weighted average by text length
        total_weighted_sentiment = sum([r['sentiment_score'] * r['text_length'] for r in results])
        total_weight = sum([r['text_length'] for r in results])
        weighted_avg_sentiment = total_weighted_sentiment / total_weight if total_weight > 0 else 0
        
        return {
            'average_sentiment': avg_sentiment,
            'weighted_average_sentiment': weighted_avg_sentiment,
            'sentiment_std': np.std([r['sentiment_score'] for r in results]),
            'total_texts': len(results),
            'positive_texts': len([r for r in results if r['sentiment_score'] > 0.1]),
            'negative_texts': len([r for r in results if r['sentiment_score'] < -0.1]),
            'neutral_texts': len([r for r in results if -0.1 <= r['sentiment_score'] <= 0.1]),
            'individual_results': results
        }

class EconomicNewsSimulator:
    """
    Simulate economic news and market sentiment data
    """
    
    def __init__(self):
        self.news_templates = [
            # Positive templates
            "Central bank announces {action} to support economic growth, markets rally",
            "GDP growth {metric} expectations, {entity} shows {trend} performance",
            "Unemployment rate {change}, signaling {sentiment} economic outlook",
            "Consumer confidence {direction}, retail sales show {trend} momentum",
            "Corporate earnings {performance}, {sector} sector leads {market_direction}",
            "Trade negotiations show {progress}, {currency} strengthens against {other_currency}",
            "Inflation remains {status}, central bank maintains {policy} monetary policy",
            "Housing market shows {condition}, construction activity {trend}",
            "Manufacturing PMI {direction}, industrial production {performance}",
            "Stock market {movement}, investor sentiment turns {mood}",
            
            # Negative templates
            "Economic concerns mount as {indicator} shows {negative_trend}",
            "Central bank warns of {risk}, {market} under pressure",
            "GDP growth {disappoints}, {entity} faces {challenge} headwinds",
            "Unemployment {rises}, labor market shows {weakness}",
            "Consumer confidence {drops}, spending patterns indicate {concern}",
            "Corporate earnings {miss} estimates, {sector} struggles with {issue}",
            "Trade tensions {escalate}, {currency} weakens amid {uncertainty}",
            "Inflation {accelerates}, central bank considers {policy_response}",
            "Housing market {softens}, construction activity {declines}",
            "Manufacturing PMI {contracts}, industrial production {falls}",
            
            # Neutral/Mixed templates
            "Economic data shows {mixed} signals as {indicator} remains {stable}",
            "Central bank maintains {current} policy stance amid {conditions}",
            "GDP growth {stable}, {entity} shows {steady} performance",
            "Labor market {holds_steady}, unemployment rate {unchanged}",
            "Consumer confidence {mixed}, spending patterns show {variation}",
            "Corporate earnings {meet} expectations, {sector} shows {steady} results",
            "Trade talks continue, {currency} trades {sideways} against {other_currency}",
            "Inflation {stable}, central bank monitors {indicators} closely",
            "Housing market {consolidates}, construction activity {steady}",
            "Manufacturing PMI {flat}, industrial production {unchanged}"
        ]
        
        self.replacement_values = {
            'action': ['rate cuts', 'stimulus measures', 'quantitative easing', 'policy support'],
            'metric': ['exceeds', 'meets', 'falls short of', 'surprises to upside'],
            'entity': ['manufacturing sector', 'services sector', 'technology sector', 'financial sector'],
            'trend': ['strong', 'robust', 'steady', 'improving', 'declining', 'weak'],
            'change': ['declines', 'falls', 'drops', 'rises', 'increases'],
            'sentiment': ['positive', 'optimistic', 'cautious', 'pessimistic'],
            'direction': ['rises', 'improves', 'strengthens', 'falls', 'weakens'],
            'performance': ['beat forecasts', 'exceed expectations', 'disappoint', 'miss estimates'],
            'sector': ['technology', 'financial', 'healthcare', 'energy', 'consumer'],
            'market_direction': ['gains', 'advances', 'rallies', 'declines', 'retreats'],
            'progress': ['positive momentum', 'breakthrough', 'setbacks', 'stalemate'],
            'currency': ['USD', 'EUR', 'GBP', 'JPY', 'CAD'],
            'other_currency': ['EUR', 'USD', 'GBP', 'JPY', 'AUD'],
            'status': ['stable', 'contained', 'elevated', 'subdued'],
            'policy': ['accommodative', 'neutral', 'restrictive', 'supportive'],
            'condition': ['strength', 'resilience', 'softness', 'weakness'],
            'movement': ['rallies', 'surges', 'advances', 'declines', 'retreats'],
            'mood': ['bullish', 'optimistic', 'bearish', 'cautious'],
            'indicator': ['employment data', 'retail sales', 'consumer spending', 'business investment'],
            'negative_trend': ['declining momentum', 'weakness', 'deterioration', 'contraction'],
            'market': ['bond markets', 'equity markets', 'currency markets', 'commodity markets'],
            'risk': ['economic risks', 'financial instability', 'market volatility', 'systemic risks'],
            'disappoints': ['disappoints', 'undershoots', 'misses forecasts'],
            'challenge': ['challenging', 'difficult', 'adverse', 'tough'],
            'weakness': ['signs of strain', 'weakness', 'softness', 'deterioration'],
            'drops': ['falls sharply', 'declines', 'drops significantly', 'weakens'],
            'concern': ['caution', 'concern', 'uncertainty', 'anxiety'],
            'miss': ['miss', 'fall short of', 'disappoint', 'undershoot'],
            'issue': ['headwinds', 'challenges', 'pressures', 'difficulties'],
            'escalate': ['intensify', 'escalate', 'worsen', 'deteriorate'],
            'uncertainty': ['uncertainty', 'volatility', 'instability', 'concerns'],
            'accelerates': ['accelerates', 'rises', 'increases', 'climbs'],
            'policy_response': ['policy tightening', 'rate hikes', 'restrictive measures'],
            'softens': ['softens', 'weakens', 'cools', 'moderates'],
            'declines': ['declines', 'falls', 'retreats', 'weakens'],
            'contracts': ['contracts', 'shrinks', 'declines', 'weakens'],
            'falls': ['falls', 'declines', 'drops', 'retreats'],
            'mixed': ['mixed', 'varied', 'divergent', 'conflicting'],
            'stable': ['stable', 'steady', 'unchanged', 'flat'],
            'current': ['current', 'existing', 'present', 'prevailing'],
            'conditions': ['conditions', 'environment', 'circumstances', 'situation'],
            'steady': ['steady', 'consistent', 'stable', 'unchanged'],
            'holds_steady': ['holds steady', 'remains stable', 'stays unchanged'],
            'unchanged': ['unchanged', 'stable', 'flat', 'steady'],
            'variation': ['variation', 'divergence', 'mixed signals', 'inconsistency'],
            'meet': ['meet', 'match', 'align with', 'come in line with'],
            'sideways': ['sideways', 'flat', 'range-bound', 'stable'],
            'indicators': ['economic indicators', 'data', 'metrics', 'signals'],
            'consolidates': ['consolidates', 'stabilizes', 'finds balance', 'holds steady'],
            'flat': ['flat', 'unchanged', 'stable', 'neutral']
        }
    
    def generate_news_headline(self):
        """
        Generate a realistic economic news headline
        """
        template = np.random.choice(self.news_templates)
        
        # Replace placeholders
        for placeholder, values in self.replacement_values.items():
            if f'{{{placeholder}}}' in template:
                replacement = np.random.choice(values)
                template = template.replace(f'{{{placeholder}}}', replacement)
        
        return template
    
    def generate_news_batch(self, n_headlines=50):
        """
        Generate a batch of economic news headlines
        """
        headlines = []
        timestamps = []
        
        # Generate headlines over the last 7 days
        end_time = datetime.now()
        start_time = end_time - timedelta(days=7)
        
        for _ in range(n_headlines):
            headline = self.generate_news_headline()
            # Random timestamp within the last week
            random_time = start_time + timedelta(
                seconds=np.random.randint(0, int((end_time - start_time).total_seconds()))
            )
            
            headlines.append(headline)
            timestamps.append(random_time)
        
        return pd.DataFrame({
            'timestamp': timestamps,
            'headline': headlines
        }).sort_values('timestamp').reset_index(drop=True)

class RealTimeMarketSentiment:
    """
    Real-time market sentiment monitoring and analysis
    """
    
    def __init__(self):
        self.sentiment_analyzer = AdvancedSentimentAnalyzer()
        self.news_simulator = EconomicNewsSimulator()
        self.sentiment_history = []
        
    def analyze_market_sentiment(self, news_data=None):
        """
        Analyze current market sentiment from news data
        """
        if news_data is None:
            # Generate simulated news data
            news_data = self.news_simulator.generate_news_batch(30)
        
        # Analyze sentiment for each headline
        sentiments = []
        for _, row in news_data.iterrows():
            sentiment_result = self.sentiment_analyzer.calculate_sentiment_score(row['headline'])
            sentiment_result['timestamp'] = row['timestamp']
            sentiment_result['headline'] = row['headline']
            sentiments.append(sentiment_result)
        
        # Create sentiment DataFrame
        sentiment_df = pd.DataFrame(sentiments)
        
        # Calculate time-weighted sentiment (recent news more important)
        now = datetime.now()
        sentiment_df['hours_ago'] = sentiment_df['timestamp'].apply(
            lambda x: (now - x).total_seconds() / 3600
        )
        sentiment_df['time_weight'] = np.exp(-sentiment_df['hours_ago'] / 24)  # Exponential decay
        
        # Weighted sentiment score
        weighted_sentiment = np.average(
            sentiment_df['sentiment_score'], 
            weights=sentiment_df['time_weight']
        )
        
        # Sentiment momentum (change over time)
        recent_sentiment = sentiment_df[sentiment_df['hours_ago'] <= 6]['sentiment_score'].mean()
        older_sentiment = sentiment_df[sentiment_df['hours_ago'] > 6]['sentiment_score'].mean()
        momentum = recent_sentiment - older_sentiment if not np.isnan(older_sentiment) else 0
        
        # Sentiment volatility
        sentiment_volatility = sentiment_df['sentiment_score'].std()
        
        # Entity-based sentiment breakdown
        entity_sentiments = {}
        for _, row in sentiment_df.iterrows():
            for entity in row['entity_details']:
                entity_name = entity['entity']
                if entity_name not in entity_sentiments:
                    entity_sentiments[entity_name] = []
                entity_sentiments[entity_name].append(row['sentiment_score'])
        
        # Average sentiment by entity
        entity_avg_sentiments = {
            entity: np.mean(scores) 
            for entity, scores in entity_sentiments.items()
        }
        
        # Overall market sentiment classification
        if weighted_sentiment > 0.2:
            market_mood = "Bullish"
        elif weighted_sentiment < -0.2:
            market_mood = "Bearish"
        else:
            market_mood = "Neutral"
        
        # Confidence level based on number of news items and consistency
        confidence = min(1.0, len(sentiment_df) / 20) * (1 - sentiment_volatility)
        confidence = max(0, min(1, confidence))
        
        result = {
            'timestamp': now,
            'weighted_sentiment': weighted_sentiment,
            'market_mood': market_mood,
            'sentiment_momentum': momentum,
            'sentiment_volatility': sentiment_volatility,
            'confidence_level': confidence,
            'total_news_items': len(sentiment_df),
            'positive_news_count': len(sentiment_df[sentiment_df['sentiment_score'] > 0.1]),
            'negative_news_count': len(sentiment_df[sentiment_df['sentiment_score'] < -0.1]),
            'neutral_news_count': len(sentiment_df[abs(sentiment_df['sentiment_score']) <= 0.1]),
            'entity_sentiments': entity_avg_sentiments,
            'recent_headlines': sentiment_df.nlargest(5, 'time_weight')[['headline', 'sentiment_score']].to_dict('records'),
            'sentiment_distribution': {
                'mean': sentiment_df['sentiment_score'].mean(),
                'std': sentiment_df['sentiment_score'].std(),
                'min': sentiment_df['sentiment_score'].min(),
                'max': sentiment_df['sentiment_score'].max(),
                'percentiles': {
                    '25th': sentiment_df['sentiment_score'].quantile(0.25),
                    '50th': sentiment_df['sentiment_score'].quantile(0.50),
                    '75th': sentiment_df['sentiment_score'].quantile(0.75)
                }
            }
        }
        
        # Store in history
        self.sentiment_history.append(result)
        
        return result
    
    def get_sentiment_trends(self, lookback_periods=10):
        """
        Analyze sentiment trends over time
        """
        if len(self.sentiment_history) < 2:
            return {'error': 'Insufficient historical data'}
        
        recent_history = self.sentiment_history[-lookback_periods:]
        
        # Extract time series
        timestamps = [h['timestamp'] for h in recent_history]
        sentiments = [h['weighted_sentiment'] for h in recent_history]
        momentum = [h['sentiment_momentum'] for h in recent_history]
        volatility = [h['sentiment_volatility'] for h in recent_history]
        
        # Calculate trends
        if len(sentiments) > 1:
            sentiment_trend = np.polyfit(range(len(sentiments)), sentiments, 1)[0]
            momentum_trend = np.polyfit(range(len(momentum)), momentum, 1)[0]
            volatility_trend = np.polyfit(range(len(volatility)), volatility, 1)[0]
        else:
            sentiment_trend = momentum_trend = volatility_trend = 0
        
        return {
            'sentiment_trend': sentiment_trend,
            'momentum_trend': momentum_trend,
            'volatility_trend': volatility_trend,
            'average_sentiment': np.mean(sentiments),
            'sentiment_volatility': np.std(sentiments),
            'periods_analyzed': len(recent_history),
            'time_range': {
                'start': timestamps[0] if timestamps else None,
                'end': timestamps[-1] if timestamps else None
            }
        }

if __name__ == "__main__":
    print("Advanced Real-Time Economic Sentiment Analysis System")
    print("=" * 55)
    
    # Initialize system
    sentiment_monitor = RealTimeMarketSentiment()
    
    # Generate and analyze sentiment
    print("Generating economic news and analyzing sentiment...")
    sentiment_result = sentiment_monitor.analyze_market_sentiment()
    
    print(f"\nMarket Sentiment Analysis Results:")
    print(f"Overall Sentiment: {sentiment_result['weighted_sentiment']:.3f}")
    print(f"Market Mood: {sentiment_result['market_mood']}")
    print(f"Confidence Level: {sentiment_result['confidence_level']:.3f}")
    print(f"Sentiment Momentum: {sentiment_result['sentiment_momentum']:.3f}")
    print(f"Total News Items: {sentiment_result['total_news_items']}")
    
    print(f"\nNews Distribution:")
    print(f"Positive: {sentiment_result['positive_news_count']}")
    print(f"Negative: {sentiment_result['negative_news_count']}")
    print(f"Neutral: {sentiment_result['neutral_news_count']}")
    
    print(f"\nTop Economic Entities by Sentiment:")
    for entity, sentiment in sorted(sentiment_result['entity_sentiments'].items(), 
                                  key=lambda x: abs(x[1]), reverse=True)[:5]:
        print(f"{entity}: {sentiment:.3f}")
    
    print("\nReal-time sentiment monitoring system ready!")
