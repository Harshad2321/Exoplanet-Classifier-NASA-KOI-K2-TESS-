"""
Quick System Validation - NASA Space Apps Challenge 2025
"""

import sys
import os
sys.path.append(os.getcwd())

def test_all():
    print("🚀 NASA Space Apps Challenge 2025 - System Validation")
    print("="*60)
    
    # Test 1: Dependencies
    print("\n1. Testing Dependencies...")
    try:
        import pandas, numpy, sklearn, matplotlib, seaborn, plotly, streamlit, xgboost, joblib
        print("   ✅ All dependencies available")
    except ImportError as e:
        print(f"   ❌ Missing dependency: {e}")
        return False
    
    # Test 2: Data files
    print("\n2. Testing Data Files...")
    try:
        import pandas as pd
        for file in ['data/raw/koi.csv', 'data/raw/k2.csv', 'data/raw/toi.csv']:
            df = pd.read_csv(file, comment='#', nrows=5, low_memory=False)
            print(f"   ✅ {file}: {len(df)} rows loaded")
    except Exception as e:
        print(f"   ❌ Data file error: {e}")
        return False
    
    # Test 3: Preprocessing outputs
    print("\n3. Testing Processed Data...")
    try:
        processed_files = ['data/processed/features.csv', 'data/processed/labels.csv']
        for file in processed_files:
            if os.path.exists(file):
                df = pd.read_csv(file, nrows=5)
                print(f"   ✅ {file}: {df.shape[0]} rows, {df.shape[1]} cols")
            else:
                print(f"   ⚠️  {file} not found (run preprocessing first)")
    except Exception as e:
        print(f"   ❌ Processed data error: {e}")
    
    # Test 4: Models
    print("\n4. Testing Trained Models...")
    try:
        import joblib
        from pathlib import Path
        model_files = list(Path('models').glob('*.joblib'))
        if model_files:
            for model_file in model_files:
                model = joblib.load(model_file)
                print(f"   ✅ {model_file.name}: Loaded successfully")
        else:
            print("   ⚠️  No trained models found (run training first)")
    except Exception as e:
        print(f"   ❌ Model loading error: {e}")
    
    # Test 5: Prediction System
    print("\n5. Testing Prediction System...")
    try:
        # Simple feature test
        sample_features = {
            'period': 365.25, 'radius': 1.0, 'temperature': 288.0,
            'insolation': 1.0, 'depth': 8400.0, 'ra': 180.0, 'dec': 0.0
        }
        
        # Try to import and create predictor
        from src.predict import ExoplanetPredictor
        
        # This may fail if model isn't trained, which is OK
        try:
            predictor = ExoplanetPredictor()
            result = predictor.predict_single(sample_features)
            print(f"   ✅ Prediction working: {result['predicted_class']} ({result['confidence']:.1%})")
        except Exception as e:
            print(f"   ⚠️  Prediction test failed: {e} (train model first)")
            
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Prediction error: {e}")
    
    # Test 6: Streamlit
    print("\n6. Testing Streamlit...")
    try:
        import streamlit as st
        print(f"   ✅ Streamlit {st.__version__} available")
        print("   📝 Launch apps with: python -m streamlit run app_simple.py")
    except Exception as e:
        print(f"   ❌ Streamlit error: {e}")
    
    print("\n" + "="*60)
    print("🎉 System validation complete!")
    print("\n📋 Next Steps:")
    print("1. If preprocessing data not found: python src/preprocess.py")
    print("2. If models not found: python src/train.py")
    print("3. Test predictions: python src/predict.py")
    print("4. Launch web app: python -m streamlit run app_simple.py")
    
    return True

if __name__ == "__main__":
    test_all()