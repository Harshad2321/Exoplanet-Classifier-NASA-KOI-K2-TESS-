#!/usr/bin/env python3
"""
🚀 NASA Space Apps Challenge 2025: Setup Script
Quick setup for the Exoplanet Hunter AI system
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command with error handling"""
    print(f"🔧 {description}...")
    try:
        subprocess.run(command, shell=True, check=True)
        print(f"✅ {description} completed!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🌌 NASA Exoplanet Hunter AI - Setup")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("nasa_app_interface.py").exists():
        print("❌ Please run this script from the project root directory")
        sys.exit(1)
    
    # Install dependencies
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        print("💡 Tip: Make sure you're in a virtual environment!")
        sys.exit(1)
    
    # Check if models exist
    if Path("nasa_models").exists() and Path("nasa_models/nasa_ensemble_model.pkl").exists():
        print("✅ NASA AI models found!")
    else:
        print("🤖 Training NASA AI models...")
        if not run_command("python nasa_clean_model.py", "Training AI models"):
            sys.exit(1)
    
    print("\n🎉 Setup complete!")
    print("🚀 Launch the NASA Exoplanet Hunter with:")
    print("   streamlit run nasa_app_interface.py")
    print("\n🌌 Ready to discover exoplanets!")

if __name__ == "__main__":
    main()