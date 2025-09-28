#!/usr/bin/env python3
"""
🚀 RTX 4060 Exoplanet Training - Simplified Version
Direct training with processed CSV data
"""

import pandas as pd
import numpy as np
import time
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import joblib

def main():
    print("🚀 RTX 4060 GPU Training - Exoplanet Classification")
    print("=" * 60)
    
    # Load processed data
    print("📥 Loading processed data...")
    try:
        X = pd.read_csv('data/processed/features.csv')
        y = pd.read_csv('data/processed/labels.csv')
        
        print(f"📊 Features: {X.shape}")
        print(f"🎯 Labels: {y.shape}")
        print(f"📋 Features: {list(X.columns)}")
        
        # Get labels as array
        y_labels = y.iloc[:, 0].values
        print(f"🏷️ Classes: {np.unique(y_labels)}")
        print(f"📈 Class counts: {pd.Series(y_labels).value_counts()}")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_labels)
    print(f"🔢 Encoded classes: {le.classes_}")
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_encoded, test_size=0.4, random_state=42, stratify=y_encoded
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"📊 Data splits:")
    print(f"   Train: {X_train.shape}")
    print(f"   Validation: {X_val.shape}")
    print(f"   Test: {X_test.shape}")
    
    # Results storage
    results = {}
    
    # 1. TensorFlow Training
    print("\n🔥 TensorFlow GPU Training")
    print("-" * 40)
    try:
        tf_result = train_tensorflow(X_train, X_val, X_test, y_train, y_val, y_test, le)
        if tf_result:
            results['TensorFlow'] = tf_result
    except Exception as e:
        print(f"⚠️ TensorFlow failed: {e}")
    
    # 2. Traditional ML Models
    print("\n🤖 Traditional ML Models")
    print("-" * 40)
    
    # Random Forest
    print("🌳 Training Random Forest...")
    start_time = time.time()
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    rf_time = time.time() - start_time
    
    joblib.dump(rf, 'models/random_forest_rtx4060.pkl')
    print(f"✅ Random Forest: {rf_acc:.4f} accuracy ({rf_time:.2f}s)")
    results['RandomForest'] = {'accuracy': rf_acc, 'time': rf_time}
    
    # XGBoost with GPU
    try:
        import xgboost as xgb
        print("🚀 Training XGBoost (GPU)...")
        start_time = time.time()
        
        # Try GPU first
        try:
            xgb_model = xgb.XGBClassifier(
                n_estimators=200,
                random_state=42,
                tree_method='gpu_hist',
                gpu_id=0,
                eval_metric='logloss'
            )
            gpu_used = True
        except:
            xgb_model = xgb.XGBClassifier(
                n_estimators=200,
                random_state=42,
                eval_metric='logloss'
            )
            gpu_used = False
        
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)
        xgb_acc = accuracy_score(y_test, xgb_pred)
        xgb_time = time.time() - start_time
        
        joblib.dump(xgb_model, 'models/xgboost_rtx4060.pkl')
        gpu_text = " (GPU)" if gpu_used else " (CPU)"
        print(f"✅ XGBoost{gpu_text}: {xgb_acc:.4f} accuracy ({xgb_time:.2f}s)")
        results['XGBoost'] = {'accuracy': xgb_acc, 'time': xgb_time}
        
    except ImportError:
        print("⚠️ XGBoost not installed")
    
    # LightGBM with GPU
    try:
        import lightgbm as lgb
        print("💡 Training LightGBM (GPU)...")
        start_time = time.time()
        
        try:
            lgb_model = lgb.LGBMClassifier(
                n_estimators=200,
                random_state=42,
                device='gpu',
                verbose=-1
            )
            gpu_used = True
        except:
            lgb_model = lgb.LGBMClassifier(
                n_estimators=200,
                random_state=42,
                verbose=-1
            )
            gpu_used = False
        
        lgb_model.fit(X_train, y_train)
        lgb_pred = lgb_model.predict(X_test)
        lgb_acc = accuracy_score(y_test, lgb_pred)
        lgb_time = time.time() - start_time
        
        joblib.dump(lgb_model, 'models/lightgbm_rtx4060.pkl')
        gpu_text = " (GPU)" if gpu_used else " (CPU)"
        print(f"✅ LightGBM{gpu_text}: {lgb_acc:.4f} accuracy ({lgb_time:.2f}s)")
        results['LightGBM'] = {'accuracy': lgb_acc, 'time': lgb_time}
        
    except ImportError:
        print("⚠️ LightGBM not installed")
    
    # Display final results
    print("\n🏆 TRAINING RESULTS")
    print("=" * 60)
    
    if results:
        for model_name, result in results.items():
            print(f"🎯 {model_name:15}: {result['accuracy']:.4f} accuracy ({result['time']:.2f}s)")
        
        # Best model
        best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
        print(f"\n🏅 Best Model: {best_model[0]} - {best_model[1]['accuracy']:.4f}")
        
        # Save results
        results_df = pd.DataFrame([
            {'Model': name, 'Accuracy': res['accuracy'], 'Time': res['time']}
            for name, res in results.items()
        ])
        results_df.to_csv('models/rtx4060_training_results.csv', index=False)
        print("💾 Results saved to models/rtx4060_training_results.csv")
        
    else:
        print("❌ No models trained successfully")
    
    print("\n✅ RTX 4060 Training Complete!")

def train_tensorflow(X_train, X_val, X_test, y_train, y_val, y_test, label_encoder):
    """Train TensorFlow model with GPU acceleration"""
    
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
        
        print("🔧 TensorFlow GPU Setup...")
        
        # GPU configuration
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.experimental.set_memory_growth(gpus[0], True)
                print(f"✅ GPU: {gpus[0].name}")
            except RuntimeError as e:
                print(f"⚠️ GPU setup warning: {e}")
        else:
            print("⚠️ No GPU found, using CPU")
        
        # Prepare data
        num_classes = len(label_encoder.classes_)
        y_train_cat = keras.utils.to_categorical(y_train, num_classes)
        y_val_cat = keras.utils.to_categorical(y_val, num_classes) 
        y_test_cat = keras.utils.to_categorical(y_test, num_classes)
        
        # Build model optimized for RTX 4060
        model = keras.Sequential([
            # Larger layers for GPU efficiency
            layers.Dense(512, activation='relu', input_shape=(X_train.shape[1],)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            
            layers.Dense(num_classes, activation='softmax')
        ])
        
        # Compile with advanced optimizer
        model.compile(
            optimizer=keras.optimizers.AdamW(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"🧠 Parameters: {model.count_params():,}")
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5),
            keras.callbacks.ModelCheckpoint('models/tensorflow_rtx4060.h5', save_best_only=True)
        ]
        
        # Train with larger batch size for GPU
        batch_size = 256 if gpus else 64
        
        print(f"🚀 Training (batch_size={batch_size})...")
        start_time = time.time()
        
        history = model.fit(
            X_train, y_train_cat,
            validation_data=(X_val, y_val_cat),
            epochs=50,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        train_time = time.time() - start_time
        
        # Evaluate
        test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
        
        print(f"✅ TensorFlow: {test_acc:.4f} accuracy ({train_time:.2f}s)")
        
        # Save
        model.save('models/tensorflow_rtx4060_final.h5')
        
        return {'accuracy': test_acc, 'time': train_time}
        
    except ImportError:
        print("⚠️ TensorFlow not available")
        return None
    except Exception as e:
        print(f"❌ TensorFlow error: {e}")
        return None

if __name__ == "__main__":
    Path("models").mkdir(exist_ok=True)
    main()