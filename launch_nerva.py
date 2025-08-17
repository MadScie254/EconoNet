"""
NERVA Launch Script
Quick deployment of the NERVA dashboard
"""

import sys
import subprocess
from pathlib import Path

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

def launch_dashboard():
    """Launch the NERVA Streamlit dashboard"""
    dashboard_path = Path(__file__).parent / "nerva" / "ui" / "dashboard.py"
    
    if not dashboard_path.exists():
        print(f"❌ Dashboard file not found: {dashboard_path}")
        return False
    
    print("🚀 Launching NERVA Dashboard...")
    print(f"📍 Dashboard URL: http://localhost:8501")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(dashboard_path),
            "--server.port", "8501",
            "--server.headless", "false",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 NERVA Dashboard stopped")
    except Exception as e:
        print(f"❌ Failed to launch dashboard: {e}")
        return False
    
    return True

def main():
    """Main launch function"""
    print("🧠 NERVA: National Economic & Risk Visual Analytics")
    print("=" * 50)
    
    # Check if we should install dependencies
    if "--install-deps" in sys.argv:
        if not install_requirements():
            sys.exit(1)
    
    # Launch dashboard
    if not launch_dashboard():
        sys.exit(1)

if __name__ == "__main__":
    main()
