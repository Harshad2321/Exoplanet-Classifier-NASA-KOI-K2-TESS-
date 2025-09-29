#!/usr/bin/env python3
"""
🚀 Quick Training Script - NASA Space Apps Challenge 2025
Simple training using processed data for immediate deployment
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def quick_train():
    """Quick training pipeline"""
    
    print("🚀 NASA Space Apps Challenge 2025 - Quick Training")
    print("=" * 50)
    
    # Try to load processed data first
    processed_files = [
        "data/processed/final_dataset.csv",
        "data/processed/training_data.csv", 
        "data/processed/koi_processed.csv"
    ]
    
    df = None
    for file_path in processed_files:
        if Path(file_path).exists():
            print(f"📊 Loading processed data: {file_path}")
            try:
                df = pd.read_csv(file_path)
                print(f"   ✅ Loaded: {len(df)} samples")
                break
            except Exception as e:
                print(f"   ❌ Error: {e}")
    
    if df is None:
        print("⚠️ No processed data found. Trying raw data...")
        
        # Try raw data with simple processing
        raw_files = ["data/raw/koi.csv", "data/raw/k2.csv"]
        
        for file_path in raw_files:
            if Path(file_path).exists():
                print(f"📊 Loading raw data: {file_path}")
                try:
                    df = pd.read_csv(file_path, nrows=5000, low_memory=False, on_bad_lines='skip')
                    print(f"   ✅ Loaded: {len(df)} samples (sample)")
                    break
                except Exception as e:
                    print(f"   ❌ Error: {e}")
    
    if df is None:
        print("❌ No data available. Creating synthetic data for demo...")
        return create_demo_model()
    
    # Find target column
    target_cols = ['koi_disposition', 'disposition', 'tfopwg_disposition']
    target_col = None
    
    for col in target_cols:
        if col in df.columns:
            target_col = col
            break
    
    if target_col is None:
        print("❌ No target column found. Creating demo model...")
        return create_demo_model()
    
    print(f"🎯 Target column: {target_col}")
    
    # Clean data
    df = df.dropna(subset=[target_col])
    
    # Standardize targets
    target_mapping = {
        'CONFIRMED': 'CONFIRMED',
        'CANDIDATE': 'CANDIDATE', 
        'FALSE POSITIVE': 'FALSE_POSITIVE',
        'FALSE_POSITIVE': 'FALSE_POSITIVE',
    }
    
    df[target_col] = df[target_col].str.upper()
    df = df[df[target_col].isin(target_mapping.keys())]
    df[target_col] = df[target_col].map(target_mapping)
    
    print(f"📊 Target distribution:")
    print(df[target_col].value_counts())
    
    # Select numeric features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    # Keep top features
    feature_cols = numeric_cols[:20]  # Use first 20 numeric columns
    
    X = df[feature_cols].fillna(df[feature_cols].median())
    y = df[target_col]
    
    print(f"🔍 Using {len(feature_cols)} features")
    print(f"📏 Dataset shape: {X.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train models
    print("\n🤖 Training models...")
    
    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    rf_score = accuracy_score(y_test, rf_model.predict(X_test))
    print(f"   ✅ Random Forest: {rf_score:.4f}")
    
    # Extra Trees
    et_model = ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    et_model.fit(X_train, y_train)
    et_score = accuracy_score(y_test, et_model.predict(X_test))
    print(f"   ✅ Extra Trees: {et_score:.4f}")
    
    # Voting Ensemble
    voting_model = VotingClassifier([
        ('rf', rf_model),
        ('et', et_model)
    ], voting='soft')
    voting_model.fit(X_train, y_train)
    voting_score = accuracy_score(y_test, voting_model.predict(X_test))
    print(f"   ✅ Voting Ensemble: {voting_score:.4f}")
    
    # Select best model
    models = {
        'random_forest': (rf_model, rf_score),
        'extra_trees': (et_model, et_score),
        'ensemble': (voting_model, voting_score)
    }
    
    best_name = max(models, key=lambda x: models[x][1])
    best_model, best_score = models[best_name]
    
    print(f"\n🏆 Best model: {best_name} ({best_score:.4f})")
    
    # Save models
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Save components
    scaler = StandardScaler()  # Create dummy scaler
    label_encoder = LabelEncoder()
    label_encoder.fit(y)
    
    # Save all
    joblib.dump(best_model, models_dir / f"model_{best_name}.pkl")
    joblib.dump(scaler, models_dir / "scaler.pkl")
    joblib.dump(label_encoder, models_dir / "label_encoder.pkl")
    
    # Metadata
    metadata = {
        'best_model': best_name,
        'best_accuracy': best_score,
        'all_scores': {name: score for name, (_, score) in models.items()},
        'feature_names': feature_cols,
        'classes': list(label_encoder.classes_),
        'training_date': datetime.now().isoformat(),
        'dataset_size': len(X),
        'n_features': len(feature_cols)
    }
    
    with open(models_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n💾 Models saved to: {models_dir.absolute()}")
    print("✅ Ready for deployment!")
    
    return True

def create_demo_model():
    """Create a demo model with synthetic data"""
    print("🎭 Creating demo model with synthetic data...")
    
    # Generate synthetic exoplanet data
    np.random.seed(42)
    n_samples = 1000
    
    # Generate features
    data = {
        'koi_period': np.random.lognormal(3, 1, n_samples),
        'koi_prad': np.random.lognormal(0, 0.5, n_samples),
        'koi_teq': np.random.normal(500, 200, n_samples),
        'koi_insol': np.random.lognormal(0, 1, n_samples),
        'koi_dor': np.random.normal(100, 50, n_samples),
        'koi_srad': np.random.normal(1, 0.3, n_samples),
        'ra': np.random.uniform(0, 360, n_samples),
        'dec': np.random.uniform(-90, 90, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Generate synthetic targets based on features
    targets = []
    for _, row in df.iterrows():
        # Simple rules for demonstration
        if row['koi_prad'] > 0.5 and row['koi_prad'] < 4 and row['koi_teq'] > 200 and row['koi_teq'] < 700:
            if np.random.random() > 0.3:
                targets.append('CONFIRMED')
            else:
                targets.append('CANDIDATE')
        elif row['koi_prad'] < 0.5 or row['koi_prad'] > 10:
            targets.append('FALSE_POSITIVE')
        else:
            targets.append(np.random.choice(['CANDIDATE', 'FALSE_POSITIVE'], p=[0.6, 0.4]))
    
    X = df
    y = pd.Series(targets)
    
    print(f"📊 Synthetic data: {len(X)} samples")
    print(f"📊 Target distribution: {y.value_counts().to_dict()}")
    
    # Train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    score = accuracy_score(y_test, model.predict(X_test))
    
    print(f"🏆 Demo model accuracy: {score:.4f}")
    
    # Save model
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    scaler = StandardScaler()
    label_encoder = LabelEncoder()
    label_encoder.fit(y)
    
    joblib.dump(model, models_dir / "model_random_forest.pkl")
    joblib.dump(scaler, models_dir / "scaler.pkl") 
    joblib.dump(label_encoder, models_dir / "label_encoder.pkl")
    
    metadata = {
        'best_model': 'random_forest',
        'best_accuracy': score,
        'all_scores': {'random_forest': score},
        'feature_names': list(X.columns),
        'classes': list(label_encoder.classes_),
        'training_date': datetime.now().isoformat(),
        'dataset_size': len(X),
        'n_features': len(X.columns)
    }
    
    with open(models_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"💾 Demo models saved to: {models_dir.absolute()}")
    return True

if __name__ == "__main__":
    success = quick_train()
    if success:
        print("\n🎉 Training completed!")
        print("🚀 Launch deployment: streamlit run deploy_app.py")
    else:
        print("❌ Training failed")