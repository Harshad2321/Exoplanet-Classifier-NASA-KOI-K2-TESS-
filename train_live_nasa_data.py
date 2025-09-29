#!/usr/bin/env python3
"""
🚀 NASA Data Downloader and Trainer - Space Apps Challenge 2025
Download fresh NASA data and train production models
"""

import pandas as pd
import numpy as np
import requests
from io import StringIO
import os
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def download_nasa_koi_data():
    """Download Kepler Objects of Interest data from NASA"""
    
    print("📡 Downloading fresh NASA KOI data...")
    
    # NASA Exoplanet Archive API URL for KOI table
    base_url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
    
    # ADQL query for KOI data with disposition
    query = """
    SELECT TOP 5000
        kepid, kepoi_name, koi_disposition, koi_score,
        koi_period, koi_prad, koi_teq, koi_insol, koi_dor,
        koi_srad, koi_smass, koi_sage, koi_steff, koi_slogg, koi_smet,
        ra, dec, koi_fpflag_nt, koi_fpflag_ss, koi_fpflag_co, koi_fpflag_ec
    FROM koi
    WHERE koi_disposition IS NOT NULL
    """
    
    params = {
        'request': 'doQuery',
        'lang': 'adql', 
        'query': query,
        'format': 'csv'
    }
    
    try:
        print("🔄 Making request to NASA Exoplanet Archive...")
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        
        # Parse CSV data
        df = pd.read_csv(StringIO(response.text))
        
        print(f"✅ Downloaded {len(df)} KOI objects from NASA")
        print(f"📊 Columns: {list(df.columns)}")
        print(f"🎯 Dispositions: {df['koi_disposition'].value_counts().to_dict()}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error downloading NASA data: {e}")
        return None

def download_confirmed_planets():
    """Download confirmed exoplanets for additional training data"""
    
    print("📡 Downloading confirmed exoplanets...")
    
    base_url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
    
    query = """
    SELECT TOP 2000
        pl_name, hostname, discoverymethod, disc_year,
        pl_orbper as period, pl_rade as prad, pl_eqt as teq,
        pl_insol as insol, pl_ratdor as dor,
        st_rad as srad, st_mass as smass, st_age as sage, 
        st_teff as steff, st_logg as slogg, st_met as smet,
        ra, dec
    FROM ps 
    WHERE pl_name IS NOT NULL 
    AND pl_orbper IS NOT NULL
    AND pl_rade IS NOT NULL
    """
    
    params = {
        'request': 'doQuery',
        'lang': 'adql',
        'query': query,
        'format': 'csv'
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        
        df = pd.read_csv(StringIO(response.text))
        df['disposition'] = 'CONFIRMED'  # All are confirmed planets
        
        print(f"✅ Downloaded {len(df)} confirmed exoplanets")
        
        return df
        
    except Exception as e:
        print(f"❌ Error downloading confirmed planets: {e}")
        return None

def create_training_dataset():
    """Create comprehensive training dataset from NASA sources"""
    
    print("🔧 Creating comprehensive training dataset...")
    
    datasets = []
    
    # Get KOI data (has all three classes)
    koi_df = download_nasa_koi_data()
    if koi_df is not None:
        # Standardize column names and add source
        koi_df = koi_df.rename(columns={
            'koi_period': 'period',
            'koi_prad': 'prad', 
            'koi_teq': 'teq',
            'koi_insol': 'insol',
            'koi_dor': 'dor',
            'koi_srad': 'srad',
            'koi_smass': 'smass',
            'koi_sage': 'sage',
            'koi_steff': 'steff',
            'koi_slogg': 'slogg',
            'koi_smet': 'smet'
        })
        koi_df['source'] = 'KOI'
        koi_df['disposition'] = koi_df['koi_disposition']
        datasets.append(koi_df)
    
    # Get confirmed planets (additional CONFIRMED examples) 
    confirmed_df = download_confirmed_planets()
    if confirmed_df is not None:
        confirmed_df['source'] = 'PS'
        datasets.append(confirmed_df)
    
    if not datasets:
        print("❌ No data downloaded successfully")
        return None
    
    # Combine datasets
    combined_df = pd.concat(datasets, ignore_index=True, sort=False)
    
    print(f"📊 Combined dataset: {len(combined_df)} total objects")
    print(f"📈 Sources: {combined_df['source'].value_counts().to_dict()}")
    print(f"🎯 Dispositions: {combined_df['disposition'].value_counts().to_dict()}")
    
    return combined_df

def train_production_model(df):
    """Train production model on NASA data"""
    
    print("\n🤖 Training Production Model on NASA Data...")
    
    # Preprocessing
    print("🔧 Preprocessing...")
    
    # Clean dispositions
    disposition_mapping = {
        'CONFIRMED': 'CONFIRMED',
        'CANDIDATE': 'CANDIDATE',
        'FALSE POSITIVE': 'FALSE_POSITIVE'
    }
    
    df = df[df['disposition'].isin(disposition_mapping.keys())]
    df['disposition'] = df['disposition'].map(disposition_mapping)
    
    print(f"📊 Final disposition distribution:")
    print(df['disposition'].value_counts())
    
    # Select features
    feature_cols = [
        'period', 'prad', 'teq', 'insol', 'dor', 
        'srad', 'smass', 'sage', 'steff', 'slogg', 'smet',
        'ra', 'dec'
    ]
    
    # Keep only available features
    available_features = [col for col in feature_cols if col in df.columns]
    print(f"🔍 Using {len(available_features)} features: {available_features}")
    
    # Create feature matrix
    X = df[available_features].copy()
    y = df['disposition'].copy()
    
    # Handle missing values
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(
        imputer.fit_transform(X),
        columns=X.columns
    )
    
    print(f"✅ Training data: {X_imputed.shape[0]} samples, {X_imputed.shape[1]} features")
    
    # Train models
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"📊 Training: {len(X_train)} samples")
    print(f"📊 Testing: {len(X_test)} samples")
    
    # Define models
    models = {
        'random_forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        'extra_trees': ExtraTreesClassifier(
            n_estimators=200,
            max_depth=15, 
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
    }
    
    # Train individual models
    trained_models = {}
    scores = {}
    
    for name, model in models.items():
        print(f"\n🔄 Training {name}...")
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"   ✅ {name}: {accuracy:.4f} accuracy")
        print(f"   📊 Classification Report:")
        print("   " + "\n   ".join(classification_report(y_test, y_pred, zero_division=0).split('\n')))
        
        trained_models[name] = model
        scores[name] = accuracy
    
    # Create ensemble
    if len(trained_models) >= 2:
        print(f"\n🗳️ Creating voting ensemble...")
        
        ensemble = VotingClassifier(
            estimators=[(name, model) for name, model in trained_models.items()],
            voting='soft'
        )
        
        ensemble.fit(X_train, y_train)
        y_pred_ensemble = ensemble.predict(X_test)
        ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)
        
        print(f"   ✅ Ensemble: {ensemble_accuracy:.4f} accuracy")
        
        trained_models['ensemble'] = ensemble
        scores['ensemble'] = ensemble_accuracy
    
    # Select best model
    best_name = max(scores, key=scores.get)
    best_score = scores[best_name]
    
    print(f"\n🏆 Best Model: {best_name} ({best_score:.4f} accuracy)")
    
    # Save models
    print("\n💾 Saving production models...")
    
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Prepare components for saving
    import joblib
    import json
    
    scaler = StandardScaler()
    label_encoder = LabelEncoder()
    label_encoder.fit(y)
    
    # Save all models
    for name, model in trained_models.items():
        model_file = models_dir / f"model_{name}.pkl"
        joblib.dump(model, model_file)
        print(f"   ✅ Saved: {model_file.name}")
    
    # Save preprocessing components
    joblib.dump(scaler, models_dir / "scaler.pkl")
    joblib.dump(label_encoder, models_dir / "label_encoder.pkl")
    joblib.dump(imputer, models_dir / "imputer.pkl")
    
    # Save metadata
    metadata = {
        'best_model': best_name,
        'best_accuracy': best_score,
        'all_scores': scores,
        'feature_names': available_features,
        'classes': list(label_encoder.classes_),
        'training_date': datetime.now().isoformat(),
        'dataset_size': len(X_imputed),
        'n_features': len(available_features),
        'data_source': 'NASA Exoplanet Archive (Live Download)'
    }
    
    with open(models_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save training data sample for reference
    sample_data = df.head(100).to_csv(index=False)
    with open(models_dir / "training_sample.csv", 'w') as f:
        f.write(sample_data)
    
    print(f"   ✅ Saved all components to: {models_dir.absolute()}")
    
    return models_dir, best_score

def main():
    """Main pipeline"""
    
    print("=" * 70)
    print("🌌 NASA SPACE APPS CHALLENGE 2025")  
    print("🚀 Live NASA Data Download & Training Pipeline")
    print("=" * 70)
    
    try:
        # Create training dataset from NASA sources
        df = create_training_dataset()
        if df is None:
            print("❌ Failed to create training dataset")
            return False
        
        # Train production model
        models_path, accuracy = train_production_model(df)
        
        print("\n" + "=" * 70)
        print("✅ NASA LIVE DATA TRAINING COMPLETED!")
        print("=" * 70)
        print(f"🏆 Best Model Accuracy: {accuracy:.4f}")
        print(f"📁 Models Location: {models_path}")
        print(f"🌐 Data Source: NASA Exoplanet Archive (Live)")
        print(f"🎯 Ready for NASA Space Apps Challenge!")
        
        print("\n🚀 Next Steps:")
        print("1. Launch deployment: streamlit run deploy_app.py")
        print("2. Test with live NASA data")
        print("3. Submit to NASA Space Apps Challenge 2025!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 SUCCESS! Ready to launch deployment interface!")
        print("🚀 Run: streamlit run deploy_app.py")
    else:
        print("\n❌ Training pipeline failed")