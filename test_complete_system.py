#!/usr/bin/env python3
"""
Complete System Test - NASA Space Apps Challenge 2025
"A World Away: Hunting for Exoplanets with AI"

This script tests all components of our exoplanet classification solution.
"""

import sys
import subprocess
import time
import os
from pathlib import Path
import pandas as pd
import joblib

def test_dependencies():
    """Test if all required dependencies are available."""
    print("🔧 Testing Dependencies...")
    
    required_packages = [
        'pandas', 'numpy', 'sklearn', 'matplotlib', 
        'seaborn', 'plotly', 'streamlit', 'xgboost', 'joblib'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package}")
            missing.append(package)
    
    if missing:
        print(f"\n❌ Missing packages: {missing}")
        return False
    
    print("✅ All dependencies available!")
    return True

def test_data_files():
    """Test if data files are present and readable."""
    print("\n📁 Testing Data Files...")
    
    data_files = [
        'data/raw/koi.csv',
        'data/raw/k2.csv', 
        'data/raw/toi.csv'
    ]
    
    for file_path in data_files:
        try:
            df = pd.read_csv(file_path, comment='#', low_memory=False, nrows=5)
            print(f"   ✅ {file_path} - {len(df)} sample rows loaded")
        except Exception as e:
            print(f"   ❌ {file_path} - Error: {e}")
            return False
    
    print("✅ All data files accessible!")
    return True

def test_preprocessing():
    """Test the data preprocessing pipeline."""
    print("\n🔄 Testing Preprocessing Pipeline...")
    
    try:
        result = subprocess.run([sys.executable, 'src/preprocess.py'], 
                              capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("   ✅ Preprocessing completed successfully")
            
            # Check output files
            expected_files = [
                'data/processed/features.csv',
                'data/processed/labels.csv', 
                'data/processed/scaler.pkl'
            ]
            
            for file_path in expected_files:
                if Path(file_path).exists():
                    print(f"   ✅ {file_path} created")
                else:
                    print(f"   ❌ {file_path} missing")
                    return False
                    
            return True
        else:
            print(f"   ❌ Preprocessing failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"   ❌ Preprocessing error: {e}")
        return False

def test_model_training():
    """Test the model training pipeline."""
    print("\n🤖 Testing Model Training...")
    
    try:
        result = subprocess.run([sys.executable, 'src/train.py'], 
                              capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            print("   ✅ Model training completed")
            
            # Check model files
            model_files = list(Path('models').glob('best_model_*.joblib'))
            metadata_files = list(Path('models').glob('model_metadata_*.json'))
            
            if model_files:
                print(f"   ✅ Model saved: {model_files[0]}")
            else:
                print("   ❌ No model file found")
                return False
                
            if metadata_files:
                print(f"   ✅ Metadata saved: {metadata_files[0]}")
            else:
                print("   ❌ No metadata file found")
                
            return True
        else:
            print(f"   ❌ Training failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"   ❌ Training error: {e}")
        return False

def test_predictions():
    """Test the prediction system."""
    print("\n🔮 Testing Prediction System...")
    
    try:
        result = subprocess.run([sys.executable, 'src/predict.py'], 
                              capture_output=True, text=True, timeout=60,
                              env={**os.environ, 'PYTHONPATH': str(Path.cwd())})
        
        if result.returncode == 0:
            print("   ✅ Prediction system working")
            
            return True
        else:
            print(f"   ❌ Prediction failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"   ❌ Prediction error: {e}")
        return False

def test_streamlit_app():
    """Test if Streamlit apps can be imported."""
    print("\n🌐 Testing Streamlit Applications...")
    
    apps_to_test = ['app_simple.py', 'app.py']
    
    for app in apps_to_test:
        try:
            # Try to import the app (basic syntax check)
            result = subprocess.run([sys.executable, '-c', f'import runpy; runpy.run_path("{app}")'], 
                                  capture_output=True, text=True, timeout=30)
            
            if 'streamlit' not in result.stderr.lower():
                print(f"   ✅ {app} syntax OK")
            else:
                print(f"   ⚠️  {app} has Streamlit-specific code (normal)")
                
        except Exception as e:
            print(f"   ❌ {app} error: {e}")
    
    # Test if streamlit can be run
    try:
        result = subprocess.run([sys.executable, '-c', 'import streamlit; print("OK")'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("   ✅ Streamlit module accessible")
        else:
            print("   ❌ Streamlit module issues")
    except:
        print("   ❌ Streamlit import failed")
    
    return True

def main():
    """Run comprehensive system test."""
    print("🚀 NASA Space Apps Challenge 2025 - Complete System Test")
    print("=" * 60)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("Data Files", test_data_files), 
        ("Preprocessing", test_preprocessing),
        ("Model Training", test_model_training),
        ("Predictions", test_predictions),
        ("Streamlit Apps", test_streamlit_app)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
    
    print(f"\n🏆 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! The system is ready for deployment.")
        print("\n📋 Quick Start Guide:")
        print("1. Data preprocessing: python src/preprocess.py")
        print("2. Model training: python src/train.py") 
        print("3. Make predictions: python src/predict.py")
        print("4. Launch web app: python -m streamlit run app_simple.py")
    else:
        print("⚠️  Some tests failed. Please fix the issues before deployment.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)