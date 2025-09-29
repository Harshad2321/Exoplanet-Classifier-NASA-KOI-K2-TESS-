#!/usr/bin/env python3
"""
🧪 NASA Space Apps Challenge 2025: Quick Test Script
Test the NASA Exoplanet Hunter AI system
"""

import sys
import traceback
from pathlib import Path

def test_model_loading():
    """Test if NASA AI models load correctly"""
    print("🔍 Testing NASA AI model loading...")
    
    try:
        import joblib
        import json
        
        model_dir = Path('nasa_models')
        
        # Test loading each component
        ensemble = joblib.load(model_dir / 'nasa_ensemble_model.pkl')
        scaler = joblib.load(model_dir / 'nasa_scaler.pkl')
        imputer = joblib.load(model_dir / 'nasa_imputer.pkl')
        encoder = joblib.load(model_dir / 'nasa_label_encoder.pkl')
        
        with open(model_dir / 'nasa_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        print(f"✅ Ensemble model: {type(ensemble).__name__}")
        print(f"✅ Scaler: {type(scaler).__name__}")
        print(f"✅ Imputer: {type(imputer).__name__}")
        print(f"✅ Encoder: {type(encoder).__name__}")
        print(f"✅ Metadata keys: {len(metadata)}")
        print(f"✅ Target classes: {metadata.get('target_classes', [])}")
        print("🚀 All NASA AI models loaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        traceback.print_exc()
        return False

def test_prediction():
    """Test a sample prediction"""
    print("\n🔮 Testing sample prediction...")
    
    try:
        # Import the NASA predictor class
        sys.path.append('.')
        
        # Create a sample input (Earth-like exoplanet)
        sample_input = {
            'koi_period': 365.25,      # Earth-like orbit
            'koi_prad': 1.0,           # Earth-sized
            'koi_teq': 288.0,          # Habitable temperature
            'koi_insol': 1.0,          # Earth-like insolation
            'koi_srad': 1.0,           # Sun-like star
            'koi_smass': 1.0,          # Sun-like mass
            'koi_steff': 5778.0,       # Sun-like temperature
            'koi_sage': 4.5,           # Sun-like age
            'koi_dor': 215.0,          # Earth-like distance ratio
            'ra': 290.0,               # Sky coordinates
            'dec': 42.0,               # Sky coordinates
            'koi_score': 0.8           # High confidence score
        }
        
        # Load the predictor class from the interface file
        with open('nasa_app_interface.py', 'r') as f:
            code = f.read()
        
        # Extract the predictor class and test it
        exec(code.replace('st.', '#st.').replace('streamlit', '#streamlit'))
        
        predictor = NASAExoplanetPredictor()
        if predictor.load_models():
            result = predictor.predict(sample_input)
            
            if result:
                print(f"✅ Prediction: {result['prediction']}")
                print(f"✅ Confidence: {result['confidence']:.1%}")
                print(f"✅ Model used: {result['model_used']}")
                print("🚀 Prediction test successful!")
                return True
            else:
                print("❌ Prediction failed")
                return False
        else:
            print("❌ Failed to load predictor")
            return False
            
    except Exception as e:
        print(f"❌ Error in prediction test: {e}")
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🌌 NASA Space Apps Challenge 2025: Exoplanet Hunter AI")
    print("=" * 60)
    print("🧪 Running system tests...\n")
    
    # Test 1: Model Loading
    test1_passed = test_model_loading()
    
    # Test 2: Prediction
    test2_passed = test_prediction()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS:")
    print(f"   🔧 Model Loading: {'✅ PASS' if test1_passed else '❌ FAIL'}")
    print(f"   🔮 Prediction: {'✅ PASS' if test2_passed else '❌ FAIL'}")
    
    if test1_passed and test2_passed:
        print("\n🏆 ALL TESTS PASSED! NASA AI system is ready! 🚀")
        print("💡 Next step: Run 'streamlit run nasa_app_interface.py'")
    else:
        print("\n⚠️ Some tests failed. Check the errors above.")
    
    print("=" * 60)

if __name__ == "__main__":
    main()