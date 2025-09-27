#!/usr/bin/env python3
"""
Test script for NASA Exoplanet Classifier

This script tests the data loading and basic functionality
with the uploaded NASA datasets.
"""

import sys
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.append('src')

def test_data_loading():
    """Test if we can load the NASA data files"""
    print("🧪 Testing Data Loading...")
    
    data_dir = Path("data/raw")
    datasets = {}
    
    # Test each dataset
    files = {
        'koi': 'koi.csv',
        'k2': 'k2.csv', 
        'toi': 'toi.csv'
    }
    
    for name, filename in files.items():
        filepath = data_dir / filename
        print(f"\n📊 Testing {name.upper()} dataset...")
        
        if filepath.exists():
            try:
                # Load with comment handling for NASA format
                df = pd.read_csv(filepath, low_memory=False, comment='#')
                datasets[name] = df
                
                print(f"  ✅ Loaded successfully: {df.shape[0]:,} rows × {df.shape[1]:,} columns")
                print(f"  📋 Sample columns: {list(df.columns[:5])}")
                
                # Check for disposition column
                if 'disposition' in df.columns:
                    print(f"  🎯 Target column found: disposition")
                    print(f"  📈 Target distribution: {df['disposition'].value_counts().to_dict()}")
                else:
                    print(f"  ⚠️ No 'disposition' column found")
                    disp_cols = [col for col in df.columns if 'disp' in col.lower()]
                    print(f"  🔍 Possible target columns: {disp_cols}")
                
            except Exception as e:
                print(f"  ❌ Failed to load: {e}")
        else:
            print(f"  ❌ File not found: {filepath}")
    
    return datasets

def test_custom_modules():
    """Test if our custom modules work"""
    print("\n🔧 Testing Custom Modules...")
    
    try:
        from data_loader import ExoplanetDataLoader
        loader = ExoplanetDataLoader()
        print("  ✅ Data loader imported successfully")
        
        # Test dataset info
        info_df = loader.get_dataset_info()
        print(f"  📋 Dataset info generated: {len(info_df)} datasets")
        
    except Exception as e:
        print(f"  ❌ Data loader failed: {e}")
    
    try:
        from preprocessing import ExoplanetPreprocessor
        preprocessor = ExoplanetPreprocessor()
        print("  ✅ Preprocessor imported successfully")
        
    except Exception as e:
        print(f"  ❌ Preprocessor failed: {e}")

def main():
    """Run all tests"""
    print("🚀 NASA Exoplanet Classifier - System Test")
    print("=" * 50)
    
    # Test data loading
    datasets = test_data_loading()
    
    # Test custom modules
    test_custom_modules()
    
    # Summary
    print(f"\n📊 Test Summary:")
    print(f"  Datasets loaded: {len(datasets)}")
    print(f"  Total records: {sum(len(df) for df in datasets.values()):,}")
    
    if datasets:
        print(f"\n✅ System ready! You can now:")
        print(f"  1. Run the Jupyter notebook: jupyter lab notebooks/01_exoplanet_classification_eda.ipynb")  
        print(f"  2. Train models: python src/train_model.py")
        print(f"  3. Launch web app: streamlit run app.py")
    else:
        print(f"\n❌ Issues found. Please check data files and try again.")

if __name__ == "__main__":
    main()