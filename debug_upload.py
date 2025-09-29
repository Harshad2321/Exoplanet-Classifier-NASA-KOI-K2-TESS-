#!/usr/bin/env python3
"""
🔧 Test CSV Upload Functionality
Debug script to identify upload errors
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys
import traceback

# Set deterministic behavior
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def test_csv_loading():
    """Test CSV loading functionality"""
    print("🧪 Testing CSV Loading...")
    
    try:
        # Test loading the sample CSV
        csv_path = "sample_test_data.csv"
        print(f"📁 Loading: {csv_path}")
        
        df = pd.read_csv(csv_path)
        print(f"✅ Successfully loaded: {len(df)} rows, {len(df.columns)} columns")
        print(f"📊 Columns: {list(df.columns)}")
        print(f"📋 Sample data:")
        print(df.head(2))
        
        return df
        
    except Exception as e:
        print(f"❌ CSV Loading Error: {e}")
        traceback.print_exc()
        return None

def test_model_loading():
    """Test model loading functionality"""
    print("\n🧪 Testing Model Loading...")
    
    try:
        models_dir = Path("models")
        
        # Check if models directory exists
        if not models_dir.exists():
            print(f"❌ Models directory not found: {models_dir}")
            return None
        
        # List available files
        model_files = list(models_dir.glob("*.pkl"))
        print(f"📁 Found {len(model_files)} .pkl files:")
        for f in model_files:
            print(f"  - {f.name}")
        
        # Try loading main components
        components = {}
        
        # Load label encoder
        le_path = models_dir / "label_encoder.pkl"
        if le_path.exists():
            components['label_encoder'] = joblib.load(le_path)
            print(f"✅ Label encoder classes: {components['label_encoder'].classes_}")
        else:
            print(f"❌ Label encoder not found: {le_path}")
        
        # Load scaler
        scaler_path = models_dir / "scaler.pkl"
        if scaler_path.exists():
            components['scaler'] = joblib.load(scaler_path)
            print(f"✅ Scaler loaded successfully")
        else:
            print(f"❌ Scaler not found: {scaler_path}")
        
        # Load random forest model
        rf_path = models_dir / "model_random_forest.pkl"
        if rf_path.exists():
            components['model'] = joblib.load(rf_path)
            print(f"✅ Random Forest model loaded")
        else:
            print(f"❌ Random Forest model not found: {rf_path}")
        
        return components
        
    except Exception as e:
        print(f"❌ Model Loading Error: {e}")
        traceback.print_exc()
        return None

def test_prediction():
    """Test prediction functionality"""
    print("\n🧪 Testing Prediction...")
    
    try:
        # Load components
        components = test_model_loading()
        if not components:
            print("❌ Cannot test prediction - models not loaded")
            return
        
        # Load test data
        df = test_csv_loading()
        if df is None:
            print("❌ Cannot test prediction - CSV not loaded")
            return
        
        # Make prediction on first row
        test_row = df.iloc[0]
        print(f"\n📊 Test sample:")
        print(test_row.to_dict())
        
        # Prepare features
        feature_cols = [col for col in df.columns if col != 'koi_disposition']
        X = test_row[feature_cols].values.reshape(1, -1)
        
        print(f"🔢 Feature shape: {X.shape}")
        print(f"🔢 Feature values: {X[0][:5]}... (showing first 5)")
        
        # Scale if scaler available
        if 'scaler' in components:
            X_scaled = components['scaler'].transform(X)
            print(f"✅ Features scaled")
        else:
            X_scaled = X
            print(f"⚠️ No scaling applied")
        
        # Make prediction
        if 'model' in components:
            prediction = components['model'].predict(X_scaled)[0]
            probabilities = components['model'].predict_proba(X_scaled)[0]
            
            # Decode prediction
            if 'label_encoder' in components:
                decoded_pred = components['label_encoder'].inverse_transform([prediction])[0]
                print(f"🎯 Prediction: {decoded_pred}")
                print(f"🎯 Probabilities: {probabilities}")
            else:
                print(f"🎯 Raw prediction: {prediction}")
                print(f"🎯 Probabilities: {probabilities}")
                
            print("✅ Prediction successful!")
        else:
            print("❌ No model available for prediction")
        
    except Exception as e:
        print(f"❌ Prediction Error: {e}")
        traceback.print_exc()

def test_robust_csv_loader():
    """Test the RobustCSVLoader functionality"""
    print("\n🧪 Testing RobustCSVLoader...")
    
    try:
        # Create a problematic CSV for testing
        problematic_csv = """koi_period,koi_prad,koi_teq,koi_insol
365.25,1.0,288.0,1.0
87.97,0.38,700.0,6.67,extra_field
225.0,0.95,462.0
100.5,2.1,350.0,3.2"""
        
        with open("test_problematic.csv", "w") as f:
            f.write(problematic_csv)
        
        print("📁 Created problematic CSV for testing")
        
        # Try to load with pandas directly
        try:
            df_direct = pd.read_csv("test_problematic.csv")
            print(f"✅ Direct pandas loading succeeded: {len(df_direct)} rows")
        except Exception as e:
            print(f"❌ Direct pandas loading failed: {e}")
        
        # Try with error handling
        try:
            df_safe = pd.read_csv("test_problematic.csv", error_bad_lines=False, warn_bad_lines=True)
            print(f"✅ Safe pandas loading succeeded: {len(df_safe)} rows")
        except Exception as e:
            print(f"❌ Safe pandas loading failed: {e}")
        
    except Exception as e:
        print(f"❌ RobustCSVLoader test error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    print("🔧 CSV Upload Error Diagnosis Tool")
    print("="*50)
    
    # Run tests
    test_csv_loading()
    test_model_loading() 
    test_prediction()
    test_robust_csv_loader()
    
    print("\n" + "="*50)
    print("🏁 Diagnosis Complete!")