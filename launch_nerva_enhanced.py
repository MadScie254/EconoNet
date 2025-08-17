"""
Enhanced NERVA Launch Script
Professional deployment with advanced features
"""

import sys
import subprocess
from pathlib import Path
import argparse

def install_requirements():
    """Install required packages"""
    requirements_file = Path(__file__).parent / "requirements_nerva.txt"
    
    if requirements_file.exists():
        print("📦 Installing NERVA dependencies...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
            ])
            print("✅ Dependencies installed successfully!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            return False
    else:
        print("❌ Requirements file not found")
        return False

def launch_enhanced_dashboard():
    """Launch the enhanced NERVA Streamlit dashboard"""
    dashboard_path = Path(__file__).parent / "nerva" / "ui" / "enhanced_dashboard.py"
    
    if not dashboard_path.exists():
        print(f"❌ Enhanced dashboard file not found: {dashboard_path}")
        return False
    
    print("🚀 Launching Enhanced NERVA Dashboard...")
    print(f"📍 Dashboard URL: http://localhost:8501")
    print("🎨 Features: FontAwesome icons, professional styling, advanced analytics")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(dashboard_path),
            "--server.port", "8501",
            "--server.headless", "false",
            "--browser.gatherUsageStats", "false",
            "--theme.base", "light",
            "--theme.primaryColor", "#667eea",
            "--theme.backgroundColor", "#ffffff",
            "--theme.secondaryBackgroundColor", "#f0f2f6"
        ])
    except KeyboardInterrupt:
        print("\n👋 Enhanced NERVA Dashboard stopped")
    except Exception as e:
        print(f"❌ Failed to launch enhanced dashboard: {e}")
        return False
    
    return True

def launch_basic_dashboard():
    """Launch the basic NERVA dashboard"""
    dashboard_path = Path(__file__).parent / "nerva" / "ui" / "dashboard.py"
    
    if not dashboard_path.exists():
        print(f"❌ Basic dashboard file not found: {dashboard_path}")
        return False
    
    print("🚀 Launching Basic NERVA Dashboard...")
    print(f"📍 Dashboard URL: http://localhost:8502")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(dashboard_path),
            "--server.port", "8502",
            "--server.headless", "false",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Basic NERVA Dashboard stopped")
    except Exception as e:
        print(f"❌ Failed to launch basic dashboard: {e}")
        return False
    
    return True

def run_notebook():
    """Launch Jupyter notebook for EDA"""
    notebook_path = Path(__file__).parent / "notebooks" / "EDA.ipynb"
    
    if not notebook_path.exists():
        print(f"❌ Notebook file not found: {notebook_path}")
        return False
    
    print("📓 Launching Jupyter Notebook for EDA...")
    
    try:
        subprocess.run([
            sys.executable, "-m", "jupyter", "notebook", str(notebook_path)
        ])
    except KeyboardInterrupt:
        print("\n👋 Jupyter Notebook stopped")
    except Exception as e:
        print(f"❌ Failed to launch notebook: {e}")
        return False
    
    return True

def run_system_test():
    """Run system validation tests"""
    test_path = Path(__file__).parent / "test_nerva.py"
    
    if not test_path.exists():
        print(f"❌ Test file not found: {test_path}")
        return False
    
    print("🧪 Running NERVA system tests...")
    
    try:
        result = subprocess.run([sys.executable, str(test_path)], capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Warnings/Errors:")
            print(result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Failed to run tests: {e}")
        return False

def main():
    """Main launcher function with command line interface"""
    parser = argparse.ArgumentParser(description="NERVA: National Economic & Risk Visual Analytics")
    parser.add_argument("--mode", "-m", choices=["enhanced", "basic", "notebook", "test"], 
                       default="enhanced", help="Launch mode")
    parser.add_argument("--install-deps", action="store_true", help="Install dependencies first")
    parser.add_argument("--test-first", action="store_true", help="Run tests before launching")
    
    args = parser.parse_args()
    
    print("🧠 NERVA: National Economic & Risk Visual Analytics")
    print("=" * 55)
    
    # Install dependencies if requested
    if args.install_deps:
        if not install_requirements():
            sys.exit(1)
        print()
    
    # Run tests if requested
    if args.test_first:
        if not run_system_test():
            print("⚠️  Tests failed, but continuing with launch...")
        print()
    
    # Launch based on mode
    success = False
    
    if args.mode == "enhanced":
        print("🎨 Launching Enhanced Dashboard with Professional UI")
        success = launch_enhanced_dashboard()
    elif args.mode == "basic":
        print("📊 Launching Basic Dashboard")
        success = launch_basic_dashboard()
    elif args.mode == "notebook":
        print("📓 Launching Jupyter Notebook for EDA")
        success = run_notebook()
    elif args.mode == "test":
        print("🧪 Running System Tests")
        success = run_system_test()
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
