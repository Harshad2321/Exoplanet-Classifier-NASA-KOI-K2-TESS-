#!/usr/bin/env python3
"""
🧪 Test the NASA Space Apps Challenge Deployment
Quick test to verify the models and deployment are working
"""

import pandas as pd
import joblib
import json
from pathlib import Path

def test_model_loading():
    """Test if models can be loaded properly"""
    
    print("🧪 Testing NASA Space Apps Challenge Models...")
    
    models_dir = Path("models")
    
    # Check if model files exist
    required_files = [
        "model_random_forest.pkl",
        "scaler.pkl", 
        "label_encoder.pkl",
        "metadata.json"
    ]
    
    print("📋 Checking required files...")
    for file in required_files:
        file_path = models_dir / file
        if file_path.exists():
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file} - MISSING")
            return False
    
    # Test loading models
    try:
        print("\n🔄 Loading models...")
        
        # Load main model
        model = joblib.load(models_dir / "model_random_forest.pkl")
        print("   ✅ Random Forest model loaded")
        
        # Load preprocessing components
        scaler = joblib.load(models_dir / "scaler.pkl")
        label_encoder = joblib.load(models_dir / "label_encoder.pkl")
        print("   ✅ Preprocessing components loaded")
        
        # Load metadata
        with open(models_dir / "metadata.json", 'r') as f:
            metadata = json.load(f)
        print("   ✅ Metadata loaded")
        
        # Print model info
        print(f"\n📊 Model Information:")
        print(f"   Best Model: {metadata.get('best_model', 'N/A')}")
        print(f"   Accuracy: {metadata.get('best_accuracy', 'N/A'):.4f}")
        print(f"   Classes: {metadata.get('classes', 'N/A')}")
        print(f"   Features: {len(metadata.get('feature_names', []))}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error loading models: {e}")
        return False

def test_prediction():
    """Test making a prediction"""
    
    print("\n🔮 Testing Prediction Pipeline...")
    
    try:
        from deploy_app import ProductionPredictor
        
        # Initialize predictor
        predictor = ProductionPredictor()
        success = predictor.load_models()
        
        if not success:
            print("   ❌ Could not load predictor")
            return False
        
        print("   ✅ Predictor loaded successfully")
        
        # Test prediction with sample data
        sample_data = {
            'period': 365.25,      # Earth-like orbit
            'prad': 1.0,           # Earth-like radius
            'teq': 288,            # Earth-like temperature
            'insol': 1.0,          # Earth-like insolation
            'dor': 215,            # Reasonable orbit
            'srad': 1.0,           # Sun-like star
            'smass': 1.0,          # Solar mass
            'sage': 4.5,           # Solar age
            'steff': 5778,         # Solar temperature
            'slogg': 4.4,          # Solar gravity
            'smet': 0.0,           # Solar metallicity
            'ra': 290.0,           # Sky position
            'dec': 42.0,           # Sky position
            'score': 0.8,          # High confidence
            'fpflag_nt': 0,        # No false positive flags
            'fpflag_ss': 0,
            'fpflag_co': 0
        }
        
        print(f"   🌍 Testing with Earth-like parameters...")
        result = predictor.predict(sample_data)
        
        if result:
            print(f"   ✅ Prediction successful!")
            print(f"      Classification: {result['prediction']}")
            print(f"      Confidence: {result['confidence']:.1%}")
            print(f"      Model: {result['model_used']}")
            return True
        else:
            print(f"   ❌ Prediction failed")
            return False
            
    except Exception as e:
        print(f"   ❌ Prediction test error: {e}")
        return False

def test_streamlit_access():
    """Test if Streamlit app is accessible"""
    
    print(f"\n🌐 Testing Streamlit App Access...")
    
    try:
        import requests
        
        # Test if the app is running
        response = requests.get("http://localhost:8501", timeout=5)
        
        if response.status_code == 200:
            print(f"   ✅ Streamlit app is running at http://localhost:8501")
            return True
        else:
            print(f"   ⚠️ Streamlit app returned status code: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"   ⚠️ Streamlit app not running or not accessible")
        print(f"   💡 Start with: streamlit run deploy_app.py")
        return False
    except Exception as e:
        print(f"   ❌ Error testing Streamlit access: {e}")
        return False

def main():
    """Run all tests"""
    
    print("=" * 60)
    print("🧪 NASA SPACE APPS CHALLENGE 2025 - SYSTEM TEST")
    print("=" * 60)
    
    tests = [
        ("Model Loading", test_model_loading),
        ("Prediction Pipeline", test_prediction),
        ("Streamlit Access", test_streamlit_access)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🔬 Running {test_name} Test...")
        success = test_func()
        results.append((test_name, success))
    
    # Summary
    print(f"\n" + "=" * 60)
    print(f"📋 TEST RESULTS SUMMARY")
    print(f"=" * 60)
    
    all_passed = True
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if not success:
            all_passed = False
    
    print(f"\n🎯 Overall Status: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    if all_passed:
        print(f"\n🚀 NASA Space Apps Challenge 2025 System Ready!")
        print(f"🌐 Web Interface: http://localhost:8501")
        print(f"📊 Upload CSV files for instant exoplanet classification")
        print(f"🌌 Ready for challenge submission!")
    else:
        print(f"\n🔧 Please fix the failed tests before deployment")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)