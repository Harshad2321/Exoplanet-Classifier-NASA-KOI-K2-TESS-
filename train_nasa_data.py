#!/usr/bin/env python3
"""
🚀 Real NASA Data Training - Space Apps Challenge 2025
Train on actual NASA exoplanet data with proper parsing
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
import joblib
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_nasa_data():
    """Load and parse NASA exoplanet data properly"""
    
    print("📊 Loading NASA Exoplanet Data...")
    
    # Try to read the KOI file with different approaches
    koi_path = "data/raw/koi.csv"
    
    # Method 1: Skip header rows that might have comments
    try:
        print("🔍 Attempting to read KOI data with comment detection...")
        
        # Read first few lines to detect header
        with open(koi_path, 'r') as f:
            lines = f.readlines()[:20]
        
        # Find the actual header row
        header_row = 0
        for i, line in enumerate(lines):
            if not line.startswith('#') and ',' in line:
                header_row = i
                break
        
        print(f"📍 Found header at row {header_row}")
        
        # Read the data
        df = pd.read_csv(koi_path, skiprows=header_row, low_memory=False, on_bad_lines='skip')
        
        print(f"✅ Successfully loaded KOI data: {df.shape}")
        print(f"📋 Columns: {list(df.columns)[:10]}...")
        
        return df
        
    except Exception as e:
        print(f"❌ Error reading KOI data: {e}")
        
        # Try K2 data instead
        k2_path = "data/raw/k2.csv"
        if Path(k2_path).exists():
            print("🔄 Trying K2 data...")
            try:
                df = pd.read_csv(k2_path, low_memory=False, on_bad_lines='skip')
                print(f"✅ Successfully loaded K2 data: {df.shape}")
                return df
            except Exception as e2:
                print(f"❌ Error reading K2 data: {e2}")
        
        return None

def preprocess_nasa_data(df):
    """Process NASA data for machine learning"""
    
    print("🔧 Preprocessing NASA data...")
    
    # Look for disposition columns
    disposition_cols = [col for col in df.columns if 'disposition' in col.lower()]
    print(f"🎯 Found disposition columns: {disposition_cols}")
    
    if not disposition_cols:
        print("❌ No disposition column found")
        return None, None
    
    target_col = disposition_cols[0]
    print(f"🎯 Using target column: {target_col}")
    
    # Clean target data
    df = df.dropna(subset=[target_col])
    df = df[df[target_col].notna()]
    
    # Show unique values
    print(f"📊 Unique dispositions: {df[target_col].unique()}")
    
    # Map disposition values to standard labels
    disposition_mapping = {
        'CONFIRMED': 'CONFIRMED',
        'CANDIDATE': 'CANDIDATE',
        'FALSE POSITIVE': 'FALSE_POSITIVE',
        'FALSE_POSITIVE': 'FALSE_POSITIVE',
        'Not Dispositioned': 'CANDIDATE',
        'DISPOSITION NOT AVAILABLE': 'CANDIDATE'
    }
    
    # Apply mapping
    df[target_col] = df[target_col].str.upper()
    df = df[df[target_col].isin(disposition_mapping.keys())]
    df[target_col] = df[target_col].map(disposition_mapping)
    
    print(f"📊 Final target distribution:")
    print(df[target_col].value_counts())
    
    # Select numerical features for exoplanet properties
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove target from features
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    # Filter out obviously non-feature columns (IDs, etc.)
    feature_cols = []
    exclude_patterns = ['id', 'rowid', 'flag', 'err', 'str']
    
    for col in numeric_cols:
        col_lower = col.lower()
        if not any(pattern in col_lower for pattern in exclude_patterns):
            feature_cols.append(col)
    
    # Limit to reasonable number of features
    feature_cols = feature_cols[:25]
    
    print(f"🔍 Selected {len(feature_cols)} features:")
    for i, col in enumerate(feature_cols[:10]):
        print(f"  {i+1}. {col}")
    if len(feature_cols) > 10:
        print(f"  ... and {len(feature_cols)-10} more")
    
    # Create feature matrix
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # Handle missing values
    print("🔧 Handling missing values...")
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(
        imputer.fit_transform(X), 
        columns=X.columns, 
        index=X.index
    )
    
    # Remove constant columns
    constant_cols = []
    for col in X_imputed.columns:
        if X_imputed[col].nunique() <= 1:
            constant_cols.append(col)
    
    if constant_cols:
        print(f"🗑️ Removing {len(constant_cols)} constant columns")
        X_imputed = X_imputed.drop(columns=constant_cols)
    
    print(f"✅ Final dataset: {X_imputed.shape[0]} samples, {X_imputed.shape[1]} features")
    
    return X_imputed, y

def train_nasa_models(X, y):
    """Train ensemble models on NASA data"""
    
    print("\n🤖 Training NASA Exoplanet Models...")
    
    # Check class balance
    class_counts = y.value_counts()
    print(f"📊 Class distribution: {class_counts.to_dict()}")
    
    # Balance dataset if needed
    min_class_size = class_counts.min()
    if min_class_size < 50:
        print("⚠️ Some classes have very few samples, using all available data")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"📊 Training set: {X_train.shape[0]} samples")
    print(f"📊 Test set: {X_test.shape[0]} samples")
    
    # Define models
    models = {
        'random_forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        'extra_trees': ExtraTreesClassifier(
            n_estimators=200, 
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
    }
    
    # Train and evaluate models
    trained_models = {}
    scores = {}
    
    for name, model in models.items():
        print(f"\n🔄 Training {name}...")
        
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"   ✅ {name}: {accuracy:.4f} accuracy")
            
            trained_models[name] = model
            scores[name] = accuracy
            
        except Exception as e:
            print(f"   ❌ {name} failed: {e}")
    
    # Create ensemble if we have multiple models
    if len(trained_models) >= 2:
        print(f"\n🗳️ Creating ensemble...")
        
        ensemble = VotingClassifier(
            estimators=[(name, model) for name, model in trained_models.items()],
            voting='soft'
        )
        
        ensemble.fit(X_train, y_train)
        ensemble_pred = ensemble.predict(X_test)
        ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
        
        print(f"   ✅ Ensemble: {ensemble_accuracy:.4f} accuracy")
        
        trained_models['ensemble'] = ensemble
        scores['ensemble'] = ensemble_accuracy
    
    # Select best model
    best_model_name = max(scores, key=scores.get)
    best_accuracy = scores[best_model_name]
    
    print(f"\n🏆 Best Model: {best_model_name} ({best_accuracy:.4f} accuracy)")
    
    return trained_models, scores, best_model_name, X.columns.tolist()

def save_nasa_models(models, scores, best_model_name, feature_names, y):
    """Save the trained models for deployment"""
    
    print("\n💾 Saving NASA Models for Deployment...")
    
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Create components
    scaler = StandardScaler()  # Placeholder
    label_encoder = LabelEncoder()
    label_encoder.fit(y)
    
    # Save all models
    for name, model in models.items():
        joblib.dump(model, models_dir / f"model_{name}.pkl")
        print(f"   ✅ Saved: model_{name}.pkl")
    
    # Save components
    joblib.dump(scaler, models_dir / "scaler.pkl")
    joblib.dump(label_encoder, models_dir / "label_encoder.pkl")
    
    # Create metadata
    metadata = {
        'best_model': best_model_name,
        'best_accuracy': scores[best_model_name],
        'all_scores': scores,
        'feature_names': feature_names,
        'classes': list(label_encoder.classes_),
        'training_date': datetime.now().isoformat(),
        'dataset_size': len(y),
        'n_features': len(feature_names),
        'data_source': 'NASA Exoplanet Archive'
    }
    
    with open(models_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"   ✅ Saved: metadata.json")
    print(f"   ✅ Saved: scaler.pkl")
    print(f"   ✅ Saved: label_encoder.pkl")
    
    # Create summary
    with open(models_dir / "model_info.txt", 'w') as f:
        f.write("🌌 NASA Exoplanet Classifier - Real Data Model\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Training Date: {metadata['training_date']}\n")
        f.write(f"Data Source: {metadata['data_source']}\n")
        f.write(f"Best Model: {metadata['best_model']}\n")
        f.write(f"Best Accuracy: {metadata['best_accuracy']:.4f}\n")
        f.write(f"Dataset Size: {metadata['dataset_size']} objects\n")
        f.write(f"Features: {metadata['n_features']}\n")
        f.write(f"Classes: {', '.join(metadata['classes'])}\n\n")
        
        f.write("Model Performance:\n")
        for model, score in scores.items():
            f.write(f"  {model}: {score:.4f}\n")
        
        f.write(f"\nFeatures Used:\n")
        for i, feature in enumerate(feature_names, 1):
            f.write(f"  {i}. {feature}\n")
    
    print(f"\n🎉 NASA Models saved successfully!")
    print(f"📁 Location: {models_dir.absolute()}")
    
    return models_dir

def main():
    """Main training pipeline"""
    
    print("=" * 60)
    print("🌌 NASA SPACE APPS CHALLENGE 2025")
    print("🚀 Real NASA Data Training Pipeline")
    print("=" * 60)
    
    try:
        # Load NASA data
        df = load_nasa_data()
        if df is None:
            print("❌ Could not load NASA data")
            return False
        
        # Preprocess data
        X, y = preprocess_nasa_data(df)
        if X is None or y is None:
            print("❌ Could not preprocess data")
            return False
        
        # Train models
        models, scores, best_name, features = train_nasa_models(X, y)
        if not models:
            print("❌ No models trained successfully")
            return False
        
        # Save for deployment
        models_path = save_nasa_models(models, scores, best_name, features, y)
        
        print("\n" + "=" * 60)
        print("✅ NASA DATA TRAINING COMPLETED!")
        print("=" * 60)
        print(f"🏆 Best Model: {best_name} ({scores[best_name]:.4f} accuracy)")
        print(f"📁 Models saved to: {models_path}")
        print(f"🚀 Ready for NASA Space Apps Challenge deployment!")
        print("\nNext steps:")
        print("1. 🌐 Launch app: streamlit run deploy_app.py")
        print("2. 🧪 Test with NASA data")
        print("3. 🎯 Submit to challenge!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Ready to deploy!")
        print("Run: streamlit run deploy_app.py")
    else:
        print("\n❌ Training failed - check the data files")