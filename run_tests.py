#!/usr/bin/env python3
"""
EconoNet CI/CD Test Runner

Simple test script that validates the core functionality and imports
without requiring external dependencies.
"""

import sys
import os
import subprocess
import importlib.util
from pathlib import Path

def test_imports():
    """Test core package imports"""
    print("🔍 Testing package imports...")
    
    # Add src to path
    current_dir = Path(__file__).parent
    sys.path.insert(0, str(current_dir / 'src'))
    
    try:
        import econonet
        print("✅ Core econonet package imports successfully")
    except ImportError as e:
        print(f"❌ Core import failed: {e}")
        return False
    
    try:
        from econonet import get_config, OperationMode, set_mode
        config = get_config()
        set_mode(OperationMode.OFFLINE)
        print("✅ Configuration system works")
    except Exception as e:
        print(f"❌ Configuration failed: {e}")
        return False
    
    try:
        from econonet import get_worldbank, get_coingecko
        print("✅ API functions import successfully")
    except ImportError as e:
        print(f"❌ API import failed: {e}")
        return False
    
    # Test optional news module
    try:
        from econonet.live_news import analyze_sentiment, generate_fallback_news
        score, label, emoji = analyze_sentiment("Great news!")
        assert label in ['bullish', 'bearish', 'neutral']
        print("✅ News module imports and works")
    except ImportError:
        print("⚠️ News module not available (optional)")
    except Exception as e:
        print(f"⚠️ News module error: {e}")
    
    return True

def test_syntax():
    """Test Python syntax of main files"""
    print("\n🔍 Testing file syntax...")
    
    files_to_check = [
        "ultra_dashboard_enhanced.py",
        "src/econonet/__init__.py",
        "src/econonet/config.py",
        "src/econonet/live_apis.py",
    ]
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            try:
                subprocess.run([sys.executable, "-m", "py_compile", file_path], 
                             check=True, capture_output=True)
                print(f"✅ {file_path} syntax OK")
            except subprocess.CalledProcessError as e:
                print(f"❌ {file_path} syntax error: {e}")
                return False
        else:
            print(f"⚠️ {file_path} not found")
    
    return True

def test_requirements():
    """Test that core requirements can be imported"""
    print("\n🔍 Testing core requirements...")
    
    core_deps = [
        'pandas',
        'numpy', 
        'plotly',
        'streamlit',
        'requests',
        'scipy',
        'sklearn'
    ]
    
    for dep in core_deps:
        try:
            __import__(dep)
            print(f"✅ {dep} available")
        except ImportError:
            print(f"❌ {dep} not available")
            return False
    
    # Test optional dependencies
    optional_deps = [
        'feedparser',
        'textblob',
        'nltk'
    ]
    
    for dep in optional_deps:
        try:
            __import__(dep)
            print(f"✅ {dep} available (optional)")
        except ImportError:
            print(f"⚠️ {dep} not available (optional)")
    
    return True

def run_basic_tests():
    """Run basic functionality tests"""
    print("\n🔍 Running basic functionality tests...")
    
    # Test fallback data generation
    try:
        from econonet.live_news import generate_fallback_news
        df = generate_fallback_news()
        assert len(df) > 0
        assert 'title' in df.columns
        print("✅ Fallback news generation works")
    except Exception as e:
        print(f"⚠️ Fallback news test failed: {e}")
    
    # Test configuration system
    try:
        from econonet import get_config, set_mode, OperationMode
        original_mode = get_config().mode
        set_mode(OperationMode.OFFLINE)
        assert get_config().mode == OperationMode.OFFLINE
        set_mode(original_mode)  # Reset
        print("✅ Configuration system works")
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False
    
    return True

def main():
    """Main test runner"""
    print("🚀 EconoNet CI/CD Test Runner")
    print("=" * 40)
    
    tests_passed = 0
    total_tests = 4
    
    # Run tests
    if test_imports():
        tests_passed += 1
    
    if test_syntax():
        tests_passed += 1
        
    if test_requirements():
        tests_passed += 1
        
    if run_basic_tests():
        tests_passed += 1
    
    # Report results
    print("\n" + "=" * 40)
    print(f"📊 Test Results: {tests_passed}/{total_tests} passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! CI/CD pipeline ready.")
        return 0
    else:
        print("❌ Some tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
