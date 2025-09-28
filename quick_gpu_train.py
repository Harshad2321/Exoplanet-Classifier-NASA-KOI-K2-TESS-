#!/usr/bin/env python3
"""
🚀 Quick GPU Training Script for RTX 4060
Uses existing processed data and creates GPU-optimized models
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# Add src to path
sys.path.append('src')

def run_quick_training():
    """Run quick training with existing data"""
    print("🚀 Starting Quick GPU Training for RTX 4060!")
    print("=" * 60)
    
    # Check for existing processed data or CSV files
    data_files = [
        'data/processed/combined_exoplanet_dataset.csv',
        'data/exoplanet_data_processed.csv',
        'data/combined_dataset.csv'
    ]
    
    data_file = None
    for file_path in data_files:
        if os.path.exists(file_path):
            data_file = file_path
            print(f"✅ Found data: {file_path}")
            break
    
    if not data_file:
        # Try to use the data loader
        try:
            from data_loader import ExoplanetDataLoader
            loader = ExoplanetDataLoader()
            print("📥 Loading datasets with data loader...")
            
            koi_data = loader.load_koi_data()
            k2_data = loader.load_k2_data() 
            tess_data = loader.load_tess_data()
            
            # Combine and save
            combined_data = pd.concat([koi_data, k2_data, tess_data], ignore_index=True)
            combined_data.to_csv('data/quick_combined_dataset.csv', index=False)
            data_file = 'data/quick_combined_dataset.csv'
            print(f"✅ Created combined dataset: {data_file}")
            
        except Exception as e:
            print(f"❌ Could not load data: {e}")
            print("💡 Please ensure data files exist in the data/ directory")
            return
    
    # Load data
    print(f"📊 Loading data from: {data_file}")
    df = pd.read_csv(data_file)
    print(f"📈 Dataset shape: {df.shape}")
    
    # Basic preprocessing
    print("🔧 Performing basic preprocessing...")
    
    # Select numeric features
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove target column if present
    target_cols = ['disposition', 'target', 'label', 'class']
    feature_columns = [col for col in numeric_columns if col.lower() not in target_cols]
    
    if not feature_columns:
        print("❌ No numeric features found!")
        return
    
    print(f"📋 Using {len(feature_columns)} features")
    
    # Prepare features and target
    X = df[feature_columns].fillna(0)  # Simple imputation
    
    # Find target column
    y_col = None
    for col in target_cols:
        if col in df.columns:
            y_col = col
            break
    
    if y_col is None:
        # Create dummy target for demo
        print("⚠️ No target column found, creating dummy classification")
        y = np.random.choice(['CONFIRMED', 'CANDIDATE', 'FALSE_POSITIVE'], size=len(X))
    else:
        y = df[y_col].fillna('FALSE_POSITIVE')
    
    # Encode target
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    print(f"🎯 Target classes: {le.classes_}")
    print(f"📊 Class distribution: {np.bincount(y_encoded)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"📊 Training set: {X_train_scaled.shape}")
    print(f"📊 Validation set: {X_val_scaled.shape}")
    print(f"📊 Test set: {X_test_scaled.shape}")
    
    # Train models
    print("\n🤖 Training Advanced Models...")
    
    # Import and train with available frameworks
    try:
        # Try TensorFlow/Keras first
        print("🔥 Training with TensorFlow...")
        train_tensorflow_models(X_train_scaled, X_val_scaled, X_test_scaled, 
                               y_train, y_val, y_test, len(le.classes_))
    except Exception as e:
        print(f"⚠️ TensorFlow training failed: {e}")
    
    try:
        # Try PyTorch
        print("🔥 Training with PyTorch...")
        train_pytorch_models(X_train_scaled, X_val_scaled, X_test_scaled,
                            y_train, y_val, y_test, len(le.classes_))
    except Exception as e:
        print(f"⚠️ PyTorch training failed: {e}")
    
    # Traditional ML models
    print("🔥 Training Traditional ML Models...")
    train_traditional_models(X_train_scaled, X_val_scaled, X_test_scaled,
                            y_train, y_val, y_test, le.classes_)
    
    print("\n✅ Training completed!")
    print("🎉 Check the models/ directory for saved models")

def train_tensorflow_models(X_train, X_val, X_test, y_train, y_val, y_test, num_classes):
    """Train TensorFlow models"""
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    
    print("📊 TensorFlow GPU Status:")
    gpus = tf.config.list_physical_devices('GPU')
    print(f"   GPUs found: {len(gpus)}")
    if gpus:
        print(f"   Using: {gpus[0].name}")
        # Enable memory growth
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except:
            pass
    
    # Convert to categorical
    y_train_cat = keras.utils.to_categorical(y_train, num_classes)
    y_val_cat = keras.utils.to_categorical(y_val, num_classes)
    y_test_cat = keras.utils.to_categorical(y_test, num_classes)
    
    # Create model
    model = keras.Sequential([
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
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"🧠 Model parameters: {model.count_params():,}")
    
    # Train
    callbacks = [
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5),
        keras.callbacks.ModelCheckpoint('models/tensorflow_gpu_model.h5', save_best_only=True)
    ]
    
    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=50,
        batch_size=256 if gpus else 64,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"🎯 TensorFlow Test Accuracy: {test_acc:.4f}")
    
    # Save model
    model.save('models/tensorflow_rtx4060_model.h5')
    print("💾 TensorFlow model saved")

def train_pytorch_models(X_train, X_val, X_test, y_train, y_val, y_test, num_classes):
    """Train PyTorch models"""
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"📊 PyTorch device: {device}")
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(device)
        X_val_tensor = torch.FloatTensor(X_val).to(device)
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        y_train_tensor = torch.LongTensor(y_train).to(device)
        y_val_tensor = torch.LongTensor(y_val).to(device)
        y_test_tensor = torch.LongTensor(y_test).to(device)
        
        # Create model
        class ExoplanetNet(nn.Module):
            def __init__(self, input_size, num_classes):
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Linear(input_size, 512),
                    nn.BatchNorm1d(512),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    
                    nn.Linear(512, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    
                    nn.Linear(256, 128),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    
                    nn.Linear(128, num_classes)
                )
            
            def forward(self, x):
                return self.layers(x)
        
        model = ExoplanetNet(X_train.shape[1], num_classes).to(device)
        print(f"🧠 PyTorch model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters())
        
        # Training loop (simplified)
        model.train()
        for epoch in range(30):
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_val_tensor)
                    val_loss = criterion(val_outputs, y_val_tensor)
                    _, predicted = torch.max(val_outputs, 1)
                    val_acc = (predicted == y_val_tensor).float().mean()
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Val Acc: {val_acc:.4f}")
                model.train()
        
        # Test evaluation
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            _, predicted = torch.max(test_outputs, 1)
            test_acc = (predicted == y_test_tensor).float().mean()
        
        print(f"🎯 PyTorch Test Accuracy: {test_acc:.4f}")
        
        # Save model
        torch.save(model.state_dict(), 'models/pytorch_rtx4060_model.pth')
        print("💾 PyTorch model saved")
        
    except ImportError:
        print("⚠️ PyTorch not available for training")

def train_traditional_models(X_train, X_val, X_test, y_train, y_val, y_test, class_names):
    """Train traditional ML models"""
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.metrics import accuracy_score, classification_report
    import joblib
    
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'GradientBoosting': GradientBoostingClassifier(random_state=42)
    }
    
    try:
        import xgboost as xgb
        models['XGBoost'] = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
    except ImportError:
        pass
    
    try:
        import lightgbm as lgb
        models['LightGBM'] = lgb.LGBMClassifier(random_state=42, verbose=-1)
    except ImportError:
        pass
    
    for name, model in models.items():
        print(f"🔥 Training {name}...")
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"🎯 {name} Test Accuracy: {acc:.4f}")
        
        # Save model
        model_path = f'models/{name.lower()}_rtx4060_model.pkl'
        joblib.dump(model, model_path)
        print(f"💾 {name} model saved to {model_path}")

if __name__ == "__main__":
    # Create directories
    Path("models").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)
    
    run_quick_training()