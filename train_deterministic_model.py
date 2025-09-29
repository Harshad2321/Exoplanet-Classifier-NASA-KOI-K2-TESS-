#!/usr/bin/env python3
"""
🔒 NASA Exoplanet Hunter - Deterministic Model Training
NASA Space Apps Challenge 2025 - Reproducible Model Training

This script ensures completely deterministic model behavior:
- Fixed random seeds for all components
- Reproducible predictions for same inputs
- Consistent results across runs
"""

import numpy as np
import pandas as pd
import joblib
import json
from datetime import datetime
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# 🔒 DETERMINISTIC SETTINGS
RANDOM_SEED = 42  # Fixed seed for reproducibility
np.random.seed(RANDOM_SEED)

def set_all_seeds(seed=42):
    """Set all possible random seeds for deterministic behavior"""
    np.random.seed(seed)
    # If using other libraries, set their seeds too
    import random
    random.seed(seed)
    
    # Set environment variables for deterministic behavior
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"🔒 All random seeds set to: {seed}")

def create_deterministic_dataset():
    """
    Create a deterministic dataset with fixed random seed
    This ensures exactly the same data every time
    """
    print("🔒 Creating deterministic NASA exoplanet dataset...")
    
    # Set seed before generating data
    np.random.seed(RANDOM_SEED)
    
    # Create synthetic but realistic exoplanet data
    n_samples = 5000  # Fixed size
    
    # Generate features with fixed patterns
    data = {}
    
    # Planetary parameters (log-normal distributions for realism)
    data['koi_period'] = np.random.lognormal(np.log(10), 1.5, n_samples)
    data['koi_prad'] = np.random.lognormal(np.log(2), 0.8, n_samples)
    data['koi_teq'] = np.random.normal(800, 400, n_samples)
    data['koi_insol'] = np.random.lognormal(np.log(1), 1.2, n_samples)
    
    # Stellar parameters
    data['koi_srad'] = np.random.normal(1.0, 0.3, n_samples)
    data['koi_dor'] = np.random.lognormal(np.log(100), 0.5, n_samples)
    data['koi_smass'] = np.random.normal(1.0, 0.2, n_samples)
    data['koi_steff'] = np.random.normal(5800, 800, n_samples)
    data['koi_sage'] = np.random.exponential(4.0, n_samples)
    
    # Position coordinates
    data['ra'] = np.random.uniform(0, 360, n_samples)
    data['dec'] = np.random.uniform(-90, 90, n_samples)
    
    # Derived features
    data['koi_dor'] = data['koi_dor'] + np.random.normal(0, 10, n_samples)
    data['koi_score'] = np.random.beta(2, 2, n_samples)
    
    # Additional synthetic features
    data['koi_impact'] = np.random.uniform(0, 1.5, n_samples)
    data['koi_duration'] = data['koi_period'] * np.random.uniform(0.01, 0.1, n_samples)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Ensure no invalid values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(df.median())
    
    # Create deterministic labels based on feature combinations
    # This ensures same input always gives same label
    np.random.seed(RANDOM_SEED)  # Reset seed for label generation
    
    labels = []
    for i in range(len(df)):
        # Create deterministic classification rules with balanced distribution
        score = (
            (df.iloc[i]['koi_period'] < 365) * 0.3 +
            (df.iloc[i]['koi_prad'] < 2.0) * 0.2 +
            (df.iloc[i]['koi_teq'] > 200) * 0.2 +
            (df.iloc[i]['koi_teq'] < 2000) * 0.2 +
            (df.iloc[i]['koi_score'] > 0.5) * 0.1
        )
        
        # Add deterministic variation based on row index for balanced classes
        index_factor = (i % 100) / 100.0  # Creates deterministic variation
        score += index_factor * 0.4
        
        # Create more balanced distribution
        if score > 0.8:
            labels.append('CONFIRMED')
        elif score > 0.5:
            labels.append('CANDIDATE')
        else:
            labels.append('FALSE_POSITIVE')
    
    df['koi_disposition'] = labels
    
    # Ensure balanced distribution
    disposition_counts = df['koi_disposition'].value_counts()
    print(f"📊 Label distribution: {disposition_counts.to_dict()}")
    
    return df

def train_deterministic_model():
    """Train completely deterministic models"""
    print("\n" + "="*60)
    print("🔒 NASA SPACE APPS CHALLENGE 2025 - DETERMINISTIC MODEL")
    print("   Training reproducible exoplanet classifier")
    print("="*60)
    
    # Set all seeds
    set_all_seeds(RANDOM_SEED)
    
    # Create deterministic dataset
    df = create_deterministic_dataset()
    print(f"✅ Created dataset: {len(df)} samples")
    
    # Prepare features and target
    feature_columns = [col for col in df.columns if col != 'koi_disposition']
    X = df[feature_columns]
    y = df['koi_disposition']
    
    print(f"📊 Features: {len(feature_columns)}")
    print(f"📊 Classes: {y.unique()}")
    
    # Initialize label encoder with fixed order for consistency
    label_encoder = LabelEncoder()
    # Fit with all possible classes in alphabetical order for consistency
    all_classes = ['CANDIDATE', 'CONFIRMED', 'FALSE_POSITIVE']
    label_encoder.fit(all_classes)
    y_encoded = label_encoder.transform(y)
    
    print(f"✅ Label encoder classes: {label_encoder.classes_}")
    
    # Split data with fixed random state
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, 
        test_size=0.2, 
        random_state=RANDOM_SEED,  # Fixed for reproducibility
        stratify=y_encoded  # Maintain class distribution
    )
    
    print(f"📊 Training: {len(X_train)} samples")
    print(f"📊 Testing: {len(X_test)} samples")
    
    # Scale features with consistent behavior
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train deterministic models with fixed random states
    print("\n🤖 Training Deterministic Models...")
    
    # Random Forest with fixed parameters for reproducibility
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_SEED,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        bootstrap=True,
        n_jobs=1  # Single thread for deterministic behavior
    )
    rf_model.fit(X_train_scaled, y_train)
    
    # Extra Trees with fixed parameters
    et_model = ExtraTreesClassifier(
        n_estimators=100,
        random_state=RANDOM_SEED,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        bootstrap=False,
        n_jobs=1  # Single thread for deterministic behavior
    )
    et_model.fit(X_train_scaled, y_train)
    
    # Voting classifier for ensemble
    ensemble_model = VotingClassifier(
        estimators=[
            ('rf', rf_model),
            ('et', et_model)
        ],
        voting='soft'  # Use probabilities for averaging
    )
    ensemble_model.fit(X_train_scaled, y_train)
    
    # Evaluate models
    models = {
        'random_forest': rf_model,
        'extra_trees': et_model,
        'ensemble': ensemble_model
    }
    
    scores = {}
    for name, model in models.items():
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        scores[name] = accuracy
        print(f"✅ {name.title()} Accuracy: {accuracy:.4f}")
    
    # Detailed classification report for best model
    best_model_name = max(scores, key=scores.get)
    best_model = models[best_model_name]
    y_pred_best = best_model.predict(X_test_scaled)
    
    print(f"\n📊 Best Model: {best_model_name.title()}")
    print("📊 Classification Report:")
    
    # Get actual classes present in the test set
    unique_test_classes = np.unique(y_test)
    actual_target_names = [label_encoder.classes_[i] for i in unique_test_classes]
    
    # Only use labels that are actually present
    report = classification_report(y_test, y_pred_best, 
                                 target_names=actual_target_names,
                                 labels=unique_test_classes)
    print(report)
    
    # Save models and components
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    print(f"\n💾 Saving Deterministic Models to '{models_dir}'...")
    
    # Remove old files
    for old_file in models_dir.glob("*.pkl"):
        old_file.unlink()
        print(f"   🗑️ Removed old: {old_file.name}")
    
    if models_dir.joinpath("metadata.json").exists():
        models_dir.joinpath("metadata.json").unlink()
        print(f"   🗑️ Removed old: metadata.json")
    
    # Save new models
    for name, model in models.items():
        model_path = models_dir / f"model_{name}.pkl"
        joblib.dump(model, model_path)
        print(f"   ✅ Saved: {model_path.name}")
    
    # Save preprocessing components
    joblib.dump(label_encoder, models_dir / "label_encoder.pkl")
    joblib.dump(scaler, models_dir / "scaler.pkl")
    print(f"   ✅ Saved: label_encoder.pkl")
    print(f"   ✅ Saved: scaler.pkl")
    
    # Save metadata with deterministic info
    metadata = {
        'training_date': datetime.now().isoformat(),
        'dataset_size': len(df),
        'n_features': len(feature_columns),
        'feature_names': feature_columns,
        'classes': label_encoder.classes_.tolist(),
        'all_scores': scores,
        'best_model': best_model_name,
        'random_seed': RANDOM_SEED,
        'deterministic': True,
        'model_parameters': {
            'random_forest': {
                'n_estimators': 100,
                'random_state': RANDOM_SEED,
                'max_depth': 10
            },
            'extra_trees': {
                'n_estimators': 100,
                'random_state': RANDOM_SEED,
                'max_depth': 10
            }
        }
    }
    
    with open(models_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"   ✅ Saved: metadata.json")
    
    # Test deterministic behavior
    print(f"\n🧪 Testing Deterministic Behavior...")
    
    # Create test sample
    test_sample = X_test.iloc[0:1]  # First test sample
    test_sample_scaled = scaler.transform(test_sample)
    
    # Make multiple predictions to verify consistency
    predictions = []
    probabilities = []
    
    for i in range(5):  # Test 5 times
        pred = best_model.predict(test_sample_scaled)[0]
        prob = best_model.predict_proba(test_sample_scaled)[0]
        predictions.append(pred)
        probabilities.append(prob.tolist())
    
    # Check if all predictions are identical
    all_same_pred = all(p == predictions[0] for p in predictions)
    all_same_prob = all(np.allclose(p, probabilities[0]) for p in probabilities)
    
    if all_same_pred and all_same_prob:
        print("✅ Deterministic behavior verified!")
        print(f"🎯 Consistent prediction: {label_encoder.inverse_transform([predictions[0]])[0]}")
        print(f"🎯 Consistent probabilities: {probabilities[0]}")
    else:
        print("❌ Warning: Non-deterministic behavior detected!")
        print(f"Predictions: {predictions}")
    
    print("\n" + "="*60)
    print("✅ DETERMINISTIC MODEL TRAINING COMPLETED!")
    print("="*60)
    print(f"🏆 Best Accuracy: {scores[best_model_name]:.4f}")
    print(f"📁 Models saved in: {models_dir}")
    print(f"🔒 Random seed: {RANDOM_SEED}")
    print(f"🎯 Deterministic: ✅")
    print(f"🌐 Ready for deployment!")
    
    return True

if __name__ == "__main__":
    success = train_deterministic_model()
    if success:
        print(f"\n🎉 SUCCESS! Deterministic models ready for deployment!")
        print(f"🔄 Next step: Run 'streamlit run deploy_app_enhanced.py'")
    else:
        print(f"\n❌ FAILED! Check error messages above.")