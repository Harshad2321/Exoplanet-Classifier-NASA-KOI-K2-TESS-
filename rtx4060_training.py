#!/usr/bin/env python3
"""
🚀 RTX 4060 Training with Existing Processed Data
Uses the already processed exoplanet data for GPU training
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys
import time

def main():
    print("🚀 RTX 4060 GPU Training with Processed Data")
    print("=" * 60)
    
    # Load processed data
    print("📥 Loading processed features and labels...")
    
    try:
        # Load features and labels
        X = pd.read_csv('data/processed/features.csv')
        y = pd.read_csv('data/processed/labels.csv')
        
        print(f"📊 Features shape: {X.shape}")
        print(f"🎯 Labels shape: {y.shape}")
        
        # Load label encoder to understand classes
        with open('data/processed/label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        
        print(f"📋 Classes: {label_encoder.classes_}")
        
    except Exception as e:
        print(f"❌ Error loading processed data: {e}")
        return
    
    # Convert to numpy arrays
    X_data = X.values
    y_data = y.values.ravel()  # Flatten if needed
    
    print(f"🔧 Data types: X={X_data.dtype}, y={y_data.dtype}")
    print(f"📈 Class distribution: {np.bincount(y_data)}")
    
    # Split data
    from sklearn.model_selection import train_test_split
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_data, y_data, test_size=0.4, random_state=42, stratify=y_data
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"📊 Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # GPU Training
    results = {}
    
    # 1. TensorFlow/Keras Training
    print("\n🔥 TensorFlow GPU Training")
    print("-" * 40)
    try:
        tf_results = train_tensorflow_gpu(X_train, X_val, X_test, y_train, y_val, y_test)
        results['TensorFlow'] = tf_results
    except Exception as e:
        print(f"⚠️ TensorFlow training failed: {e}")
    
    # 2. Traditional ML with GPU acceleration where possible
    print("\n🔥 Traditional ML Training (GPU accelerated)")
    print("-" * 50)
    try:
        ml_results = train_traditional_gpu(X_train, X_val, X_test, y_train, y_val, y_test)
        results.update(ml_results)
    except Exception as e:
        print(f"⚠️ Traditional ML training failed: {e}")
    
    # 3. Display Results
    print("\n🏆 FINAL RESULTS")
    print("=" * 60)
    
    for model_name, result in results.items():
        if result:
            print(f"🎯 {model_name}: {result['accuracy']:.4f} accuracy ({result['time']:.2f}s)")
    
    if results:
        best_model = max(results.items(), key=lambda x: x[1]['accuracy'] if x[1] else 0)
        print(f"\n🏅 Best Model: {best_model[0]} - {best_model[1]['accuracy']:.4f}")
    
    print("\n✅ GPU Training Complete!")
    print("💾 Models saved in models/ directory")

def train_tensorflow_gpu(X_train, X_val, X_test, y_train, y_val, y_test):
    """Train TensorFlow model with GPU acceleration"""
    
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
        
        # GPU Setup
        print("🔧 Configuring TensorFlow for RTX 4060...")
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # Enable memory growth
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"✅ GPU configured: {gpus[0].name}")
            except RuntimeError as e:
                print(f"⚠️ GPU setup warning: {e}")
        else:
            print("⚠️ No GPU found, using CPU")
        
        # Prepare data
        num_classes = len(np.unique(y_train))
        y_train_cat = keras.utils.to_categorical(y_train, num_classes)
        y_val_cat = keras.utils.to_categorical(y_val, num_classes)
        y_test_cat = keras.utils.to_categorical(y_test, num_classes)
        
        # Enhanced model architecture for RTX 4060
        model = keras.Sequential([
            # Input layer
            layers.Dense(1024, activation='relu', input_shape=(X_train.shape[1],)),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            
            # Hidden layers - larger for GPU efficiency
            layers.Dense(512, activation='relu'),
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
            
            # Output layer
            layers.Dense(num_classes, activation='softmax')
        ])
        
        # Compile with advanced optimizer
        model.compile(
            optimizer=keras.optimizers.AdamW(learning_rate=0.001, weight_decay=1e-4),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        print(f"🧠 Model parameters: {model.count_params():,}")
        
        # Enhanced callbacks for GPU training
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_accuracy',
                factor=0.5,
                patience=8,
                min_lr=1e-6,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                'models/tensorflow_rtx4060_best.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train with larger batch size for GPU efficiency
        batch_size = 512 if gpus else 128
        
        print(f"🚀 Training with batch size {batch_size}...")
        start_time = time.time()
        
        history = model.fit(
            X_train, y_train_cat,
            validation_data=(X_val, y_val_cat),
            epochs=100,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        training_time = time.time() - start_time
        
        # Evaluate
        test_results = model.evaluate(X_test, y_test_cat, verbose=0)
        test_acc = test_results[1]  # accuracy is second metric
        
        print(f"🎯 TensorFlow Test Accuracy: {test_acc:.4f}")
        print(f"⏱️ Training Time: {training_time:.2f} seconds")
        
        # Save final model
        model.save('models/tensorflow_rtx4060_final.h5')
        
        return {
            'accuracy': test_acc,
            'time': training_time,
            'model_params': model.count_params()
        }
        
    except Exception as e:
        print(f"❌ TensorFlow training error: {e}")
        return None

def train_traditional_gpu(X_train, X_val, X_test, y_train, y_val, y_test):
    """Train traditional ML models with GPU acceleration where possible"""
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    import joblib
    
    results = {}
    
    # Random Forest (CPU but optimized)
    print("🌳 Training Random Forest...")
    start_time = time.time()
    
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        random_state=42,
        n_jobs=-1  # Use all CPU cores
    )
    
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    rf_time = time.time() - start_time
    
    joblib.dump(rf, 'models/random_forest_rtx4060.pkl')
    
    print(f"🎯 Random Forest Accuracy: {rf_acc:.4f} ({rf_time:.2f}s)")
    results['RandomForest'] = {'accuracy': rf_acc, 'time': rf_time}
    
    # XGBoost with GPU support
    try:
        import xgboost as xgb
        print("🚀 Training XGBoost with GPU acceleration...")
        start_time = time.time()
        
        # Try GPU training first, fall back to CPU
        try:
            xgb_model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                random_state=42,
                tree_method='gpu_hist',  # GPU acceleration
                gpu_id=0,
                eval_metric='logloss'
            )
        except:
            # Fallback to CPU
            xgb_model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss'
            )
        
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)
        xgb_acc = accuracy_score(y_test, xgb_pred)
        xgb_time = time.time() - start_time
        
        joblib.dump(xgb_model, 'models/xgboost_rtx4060.pkl')
        
        print(f"🎯 XGBoost Accuracy: {xgb_acc:.4f} ({xgb_time:.2f}s)")
        results['XGBoost'] = {'accuracy': xgb_acc, 'time': xgb_time}
        
    except ImportError:
        print("⚠️ XGBoost not available")
    except Exception as e:
        print(f"⚠️ XGBoost training failed: {e}")
    
    # LightGBM with GPU support
    try:
        import lightgbm as lgb
        print("💡 Training LightGBM with GPU acceleration...")
        start_time = time.time()
        
        # Try GPU first, fall back to CPU
        try:
            lgb_model = lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                random_state=42,
                device='gpu',
                gpu_platform_id=0,
                gpu_device_id=0,
                verbose=-1
            )
        except:
            # Fallback to CPU
            lgb_model = lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                random_state=42,
                verbose=-1
            )
        
        lgb_model.fit(X_train, y_train)
        lgb_pred = lgb_model.predict(X_test)
        lgb_acc = accuracy_score(y_test, lgb_pred)
        lgb_time = time.time() - start_time
        
        joblib.dump(lgb_model, 'models/lightgbm_rtx4060.pkl')
        
        print(f"🎯 LightGBM Accuracy: {lgb_acc:.4f} ({lgb_time:.2f}s)")
        results['LightGBM'] = {'accuracy': lgb_acc, 'time': lgb_time}
        
    except ImportError:
        print("⚠️ LightGBM not available")
    except Exception as e:
        print(f"⚠️ LightGBM training failed: {e}")
    
    return results

if __name__ == "__main__":
    # Ensure models directory exists
    Path("models").mkdir(exist_ok=True)
    main()