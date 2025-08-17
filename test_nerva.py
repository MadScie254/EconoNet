"""
NERVA System Test
Quick validation of core functionality
"""

import sys
from pathlib import Path

# Add NERVA to path
nerva_path = Path(__file__).parent / "nerva"
sys.path.insert(0, str(nerva_path))

def test_configuration():
    """Test configuration loading"""
    print("🔧 Testing Configuration...")
    try:
        from config.settings import config
        print(f"✅ Config loaded successfully")
        print(f"   - Data path: {config.data.raw_data_path}")
        print(f"   - CBK files: {len(config.data.cbk_data_files)} configured")
        return True
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False

def test_data_processing():
    """Test data processing"""
    print("\n📊 Testing Data Processing...")
    try:
        from etl.processor import CBKDataProcessor
        
        processor = CBKDataProcessor()
        print(f"✅ ETL processor initialized")
        
        # Check if raw data exists
        raw_files = list(processor.raw_path.glob("*.csv"))
        print(f"   - Found {len(raw_files)} CSV files in raw data")
        
        if len(raw_files) > 0:
            # Test loading first file
            test_file = raw_files[0]
            df = processor._load_file(test_file)
            if df is not None:
                print(f"✅ Successfully loaded {test_file.name}: {df.shape}")
            else:
                print(f"⚠️  Could not load {test_file.name}")
        
        return True
    except Exception as e:
        print(f"❌ Data processing test failed: {e}")
        return False

def test_baseline_models():
    """Test baseline model initialization"""
    print("\n🤖 Testing Baseline Models...")
    try:
        from models.baseline import BaselineForecaster
        
        forecaster = BaselineForecaster()
        print(f"✅ Baseline forecaster initialized")
        print(f"   - Configured horizons: {forecaster.horizons}")
        return True
    except Exception as e:
        print(f"❌ Baseline models test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧠 NERVA System Validation")
    print("=" * 40)
    
    tests = [
        test_configuration,
        test_data_processing, 
        test_baseline_models
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n🎯 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🚀 NERVA system ready for launch!")
        print("Run: python launch_nerva.py")
    else:
        print("⚠️  Some tests failed. Check dependencies and data paths.")

if __name__ == "__main__":
    main()
