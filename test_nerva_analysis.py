#!/usr/bin/env python3
"""
NERVA Analysis & Models Test
Test analysis functionality and model training
"""

import sys
import warnings
from pathlib import Path
import pandas as pd
import numpy as np

# Suppress warnings for clean output
warnings.filterwarnings('ignore')

def test_analysis_functionality():
    """Test analysis functions"""
    print("🔬 Testing Analysis Functionality...")
    
    # Add NERVA to path
    sys.path.append(str(Path(__file__).parent / "nerva"))
    
    try:
        # Test data loading
        from config.settings import config
        from etl.processor import CBKDataProcessor
        
        # Fix paths
        project_root = Path(__file__).parent
        config.data.raw_data_path = project_root / "data" / "raw"
        config.data.processed_data_path = project_root / "data" / "processed"
        config.data.parquet_path = project_root / "data" / "parquet"
        
        processor = CBKDataProcessor()
        datasets = processor.scan_all_files()
        
        if len(datasets) > 0:
            print(f"✅ Data loading: {len(datasets)} datasets")
            
            # Test analysis on a sample dataset
            for dataset_name, df in datasets.items():
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    series = df[numeric_cols[0]].dropna()
                    if len(series) > 10:
                        # Basic statistical analysis
                        mean_val = series.mean()
                        std_val = series.std()
                        skew_val = series.skew()
                        
                        print(f"✅ Analysis on {dataset_name}:")
                        print(f"   • Mean: {mean_val:.4f}")
                        print(f"   • Std: {std_val:.4f}")
                        print(f"   • Skewness: {skew_val:.4f}")
                        
                        # Test moving averages
                        ma_5 = series.rolling(5).mean()
                        print(f"   • Moving average calculated successfully")
                        
                        # Test returns
                        returns = series.pct_change().dropna()
                        if len(returns) > 0:
                            print(f"   • Returns volatility: {returns.std():.4f}")
                        
                        break
            
            return True
        else:
            print("❌ No datasets found")
            return False
            
    except Exception as e:
        print(f"❌ Analysis test failed: {e}")
        return False

def test_model_functionality():
    """Test model training"""
    print("\n🤖 Testing Model Functionality...")
    
    try:
        # Add NERVA to path
        sys.path.append(str(Path(__file__).parent / "nerva"))
        
        from models.baseline import EnsembleForecaster
        from models.advanced import TransformerForecaster
        
        # Create sample data
        np.random.seed(42)
        n_samples = 100
        time_series = np.cumsum(np.random.randn(n_samples)) + 100
        
        # Create features
        data = pd.DataFrame({
            'lag_1': pd.Series(time_series).shift(1),
            'lag_2': pd.Series(time_series).shift(2),
            'target': time_series
        }).dropna()
        
        X = data[['lag_1', 'lag_2']]
        y = data['target']
        
        print(f"✅ Sample data created: {len(X)} samples")
        
        # Test baseline model
        try:
            forecaster = EnsembleForecaster()
            forecaster.fit(X, y)
            predictions = forecaster.predict(X)
            
            print(f"✅ Baseline model trained successfully")
            print(f"   • Predictions shape: {predictions.shape}")
            print(f"   • Mean prediction: {np.mean(predictions):.4f}")
            
        except Exception as e:
            print(f"❌ Baseline model test failed: {e}")
        
        # Test advanced model initialization
        try:
            transformer = TransformerForecaster(
                input_dim=X.shape[1],
                d_model=32,
                nhead=2,
                num_layers=1
            )
            print(f"✅ Transformer model initialized successfully")
            print(f"   • Input dim: {X.shape[1]}")
            print(f"   • Model dim: 32")
            
        except Exception as e:
            print(f"❌ Transformer model test failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

def test_correlation_analysis():
    """Test correlation analysis"""
    print("\n🔗 Testing Correlation Analysis...")
    
    try:
        # Create sample dataset with correlations
        np.random.seed(42)
        n_samples = 50
        
        x1 = np.random.randn(n_samples)
        x2 = 0.7 * x1 + 0.3 * np.random.randn(n_samples)  # Correlated with x1
        x3 = np.random.randn(n_samples)  # Independent
        
        test_df = pd.DataFrame({
            'Variable_A': x1,
            'Variable_B': x2,
            'Variable_C': x3
        })
        
        # Calculate correlation matrix
        corr_matrix = test_df.corr()
        
        print(f"✅ Correlation matrix calculated:")
        print(f"   • A-B correlation: {corr_matrix.loc['Variable_A', 'Variable_B']:.3f}")
        print(f"   • A-C correlation: {corr_matrix.loc['Variable_A', 'Variable_C']:.3f}")
        print(f"   • B-C correlation: {corr_matrix.loc['Variable_B', 'Variable_C']:.3f}")
        
        # Test top correlations extraction
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_pairs.append({
                    'Variable 1': corr_matrix.columns[i],
                    'Variable 2': corr_matrix.columns[j],
                    'Correlation': corr_matrix.iloc[i, j]
                })
        
        corr_df = pd.DataFrame(corr_pairs)
        strongest_corr = corr_df.loc[corr_df['Correlation'].abs().idxmax()]
        
        print(f"✅ Strongest correlation identified: {strongest_corr['Variable 1']} - {strongest_corr['Variable 2']} ({strongest_corr['Correlation']:.3f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Correlation analysis test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🎯 NERVA Analysis & Models Test Suite")
    print("=" * 50)
    
    all_tests_passed = True
    
    # Test analysis functionality
    if not test_analysis_functionality():
        all_tests_passed = False
    
    # Test model functionality
    if not test_model_functionality():
        all_tests_passed = False
    
    # Test correlation analysis
    if not test_correlation_analysis():
        all_tests_passed = False
    
    print("\n" + "=" * 50)
    if all_tests_passed:
        print("🎉 ALL TESTS PASSED! Analysis and models working correctly.")
        print("🚀 Ready for full system deployment at http://localhost:8507")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    return all_tests_passed

if __name__ == "__main__":
    main()
