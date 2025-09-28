"""
🚀 Advanced GPU-Accelerated Exoplanet Model Training System
NASA Space Apps Challenge 2025 - High-Performance ML

This module implements advanced machine learning approaches:
- GPU-accelerated XGBoost and LightGBM training
- Advanced ensemble methods with stacking and voting
- Bayesian optimization for hyperparameter tuning
- Feature engineering and selection
- Cross-validation with advanced metrics
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.pipeline import Pipeline
import optuna
from pathlib import Path
import json
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import time
from datetime import datetime
import joblib
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import SMOTE, ADASYN
import catboost as cb

warnings.filterwarnings('ignore')

class AdvancedGPUExoplanetTrainer:
    """
    Advanced GPU-accelerated machine learning trainer for exoplanet classification
    """
    
    def __init__(self, data_dir="data", models_dir="models", results_dir="reports"):
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.results_dir = Path(results_dir)
        
        # Create directories
        self.models_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.scaler = None
        self.label_encoder = None
        self.feature_selector = None
        self.models = {}
        self.results = {}
        
        # GPU configuration
        self.use_gpu = True
        self.gpu_id = 0
        
        print("🚀 Advanced GPU-Accelerated Exoplanet Trainer Initialized!")
        print(f"📁 Models directory: {self.models_dir}")
        print(f"📊 Results directory: {self.results_dir}")
    
    def load_data(self):
        """Load and combine all exoplanet datasets"""
        print("\n📥 Loading exoplanet datasets...")
        
        datasets = []
        
        # Load Kepler data
        kepler_path = self.data_dir / "processed" / "kepler_processed.csv"
        if kepler_path.exists():
            kepler_df = pd.read_csv(kepler_path)
            kepler_df['source'] = 'kepler'
            datasets.append(kepler_df)
            print(f"✅ Kepler data: {len(kepler_df)} samples")
        
        # Load K2 data
        k2_path = self.data_dir / "processed" / "k2_processed.csv"
        if k2_path.exists():
            k2_df = pd.read_csv(k2_path)
            k2_df['source'] = 'k2'
            datasets.append(k2_df)
            print(f"✅ K2 data: {len(k2_df)} samples")
        
        # Load TESS data
        tess_path = self.data_dir / "processed" / "tess_processed.csv"
        if tess_path.exists():
            tess_df = pd.read_csv(tess_path)
            tess_df['source'] = 'tess'
            datasets.append(tess_df)
            print(f"✅ TESS data: {len(tess_df)} samples")
        
        if not datasets:
            raise FileNotFoundError("No processed datasets found!")
        
        # Combine datasets
        combined_df = pd.concat(datasets, ignore_index=True)
        print(f"🔗 Combined dataset: {len(combined_df)} total samples")
        
        # Display class distribution
        print("\n📊 Class distribution:")
        class_dist = combined_df['disposition'].value_counts()
        for class_name, count in class_dist.items():
            print(f"  {class_name}: {count} ({count/len(combined_df)*100:.1f}%)")
        
        return combined_df
    
    def engineer_features(self, df):
        """Advanced feature engineering with astronomical domain knowledge"""
        print("\n🔧 Engineering advanced features...")
        
        feature_df = df.copy()
        
        # Basic features
        base_features = [
            'period', 'duration', 'depth', 'prad', 'sma', 'teq', 'insol',
            'steff', 'slogg', 'srad', 'smass'
        ]
        
        # Remove rows with too many missing values
        feature_df = feature_df.dropna(subset=base_features, thresh=len(base_features)//2)
        
        # Fill remaining missing values
        for col in base_features:
            if col in feature_df.columns:
                feature_df[col] = feature_df[col].fillna(feature_df[col].median())
        
        # Advanced derived features
        if 'period' in feature_df.columns:
            feature_df['period_log'] = np.log10(feature_df['period'].clip(lower=1e-6))
            feature_df['period_sqrt'] = np.sqrt(feature_df['period'])
        
        if 'duration' in feature_df.columns:
            feature_df['duration_log'] = np.log10(feature_df['duration'].clip(lower=1e-6))
        
        if 'depth' in feature_df.columns:
            feature_df['depth_log'] = np.log10(feature_df['depth'].clip(lower=1e-9))
            feature_df['depth_sqrt'] = np.sqrt(feature_df['depth'])
        
        # Planetary characteristics
        if 'prad' in feature_df.columns and 'period' in feature_df.columns:
            feature_df['density_proxy'] = feature_df['prad'] / (feature_df['period'] ** (2/3))
        
        if 'teq' in feature_df.columns and 'steff' in feature_df.columns:
            feature_df['temp_ratio'] = feature_df['teq'] / feature_df['steff'].clip(lower=1000)
        
        if 'insol' in feature_df.columns:
            feature_df['insol_log'] = np.log10(feature_df['insol'].clip(lower=1e-6))
        
        # Stellar characteristics
        if 'srad' in feature_df.columns and 'smass' in feature_df.columns:
            feature_df['stellar_density'] = feature_df['smass'] / (feature_df['srad'] ** 3)
        
        if 'steff' in feature_df.columns:
            feature_df['steff_scaled'] = (feature_df['steff'] - 5778) / 1000  # Solar temperature normalized
        
        # Transit characteristics
        if 'duration' in feature_df.columns and 'period' in feature_df.columns:
            feature_df['duration_period_ratio'] = feature_df['duration'] / (feature_df['period'] * 24)
        
        # Source encoding
        source_encoded = pd.get_dummies(feature_df['source'], prefix='source')
        feature_df = pd.concat([feature_df, source_encoded], axis=1)
        
        print(f"✅ Feature engineering complete: {feature_df.shape[1]} features")
        
        return feature_df
    
    def prepare_features(self, df):
        """Prepare features for training"""
        print("\n🎯 Preparing features for training...")
        
        # Select numeric features
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Exclude target and unnecessary columns
        exclude_cols = ['disposition', 'source']
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        X = df[feature_cols].copy()
        y = df['disposition'].copy()
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Remove infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        print(f"✅ Features prepared: {X.shape}")
        print(f"✅ Target classes: {y.value_counts().to_dict()}")
        
        return X, y
    
    def optimize_hyperparameters(self, X_train, y_train, model_name, n_trials=100):
        """Bayesian optimization for hyperparameters using Optuna"""
        print(f"\n🎯 Optimizing hyperparameters for {model_name}...")
        
        def objective(trial):
            if model_name == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                    'tree_method': 'gpu_hist' if self.use_gpu else 'hist',
                    'gpu_id': self.gpu_id if self.use_gpu else None,
                    'random_state': 42
                }
                model = xgb.XGBClassifier(**params)
            
            elif model_name == 'lightgbm':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                    'device_type': 'gpu' if self.use_gpu else 'cpu',
                    'gpu_device_id': self.gpu_id if self.use_gpu else -1,
                    'random_state': 42
                }
                model = lgb.LGBMClassifier(**params)
            
            elif model_name == 'catboost':
                params = {
                    'iterations': trial.suggest_int('iterations', 100, 1000),
                    'depth': trial.suggest_int('depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0),
                    'task_type': 'GPU' if self.use_gpu else 'CPU',
                    'devices': f'0:{self.gpu_id}' if self.use_gpu else None,
                    'random_state': 42,
                    'verbose': False
                }
                model = cb.CatBoostClassifier(**params)
            
            # Cross-validation
            cv_scores = cross_val_score(
                model, X_train, y_train, 
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                scoring='accuracy', n_jobs=-1
            )
            
            return cv_scores.mean()
        
        # Create study
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, timeout=1800)  # 30 minute timeout
        
        print(f"✅ Best {model_name} parameters: {study.best_params}")
        print(f"✅ Best {model_name} score: {study.best_value:.4f}")
        
        return study.best_params
    
    def train_advanced_models(self, X_train, y_train, X_val, y_val):
        """Train multiple advanced models with GPU acceleration"""
        print("\n🚀 Training advanced models with GPU acceleration...")
        
        models_config = {
            'xgboost_gpu': {
                'model': xgb.XGBClassifier,
                'params': {
                    'n_estimators': 500,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'tree_method': 'gpu_hist' if self.use_gpu else 'hist',
                    'gpu_id': self.gpu_id if self.use_gpu else None,
                    'random_state': 42
                }
            },
            'lightgbm_gpu': {
                'model': lgb.LGBMClassifier,
                'params': {
                    'n_estimators': 500,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'device_type': 'gpu' if self.use_gpu else 'cpu',
                    'gpu_device_id': self.gpu_id if self.use_gpu else -1,
                    'random_state': 42
                }
            },
            'catboost_gpu': {
                'model': cb.CatBoostClassifier,
                'params': {
                    'iterations': 500,
                    'depth': 6,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'task_type': 'GPU' if self.use_gpu else 'CPU',
                    'devices': f'0:{self.gpu_id}' if self.use_gpu else None,
                    'random_state': 42,
                    'verbose': False
                }
            },
            'random_forest': {
                'model': RandomForestClassifier,
                'params': {
                    'n_estimators': 500,
                    'max_depth': 10,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'random_state': 42,
                    'n_jobs': -1
                }
            },
            'extra_trees': {
                'model': ExtraTreesClassifier,
                'params': {
                    'n_estimators': 500,
                    'max_depth': 10,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'random_state': 42,
                    'n_jobs': -1
                }
            }
        }
        
        trained_models = {}
        training_results = {}
        
        for name, config in models_config.items():
            print(f"\n🔄 Training {name}...")
            start_time = time.time()
            
            try:
                # Initialize model
                model = config['model'](**config['params'])
                
                # Train model
                model.fit(X_train, y_train)
                
                # Predictions
                train_pred = model.predict(X_train)
                val_pred = model.predict(X_val)
                
                # Calculate metrics
                train_acc = accuracy_score(y_train, train_pred)
                val_acc = accuracy_score(y_val, val_pred)
                
                training_time = time.time() - start_time
                
                # Store results
                trained_models[name] = model
                training_results[name] = {
                    'train_accuracy': train_acc,
                    'val_accuracy': val_acc,
                    'training_time': training_time,
                    'params': config['params']
                }
                
                print(f"✅ {name}: Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Time: {training_time:.2f}s")
                
                # Save model
                model_path = self.models_dir / f"{name}_advanced.pkl"
                joblib.dump(model, model_path)
                
            except Exception as e:
                print(f"❌ Error training {name}: {str(e)}")
                continue
        
        return trained_models, training_results
    
    def create_ensemble(self, models, X_train, y_train, X_val, y_val):
        """Create advanced ensemble models"""
        print("\n🎭 Creating advanced ensemble models...")
        
        from sklearn.ensemble import VotingClassifier, StackingClassifier
        
        # Prepare base models for ensemble
        base_models = [(name, model) for name, model in models.items()]
        
        ensemble_models = {}
        
        # Voting classifier
        print("🔄 Training Voting Ensemble...")
        voting_clf = VotingClassifier(estimators=base_models, voting='soft', n_jobs=-1)
        voting_clf.fit(X_train, y_train)
        
        val_pred = voting_clf.predict(X_val)
        val_acc = accuracy_score(y_val, val_pred)
        
        ensemble_models['voting_ensemble'] = voting_clf
        print(f"✅ Voting Ensemble - Val Acc: {val_acc:.4f}")
        
        # Stacking classifier
        print("🔄 Training Stacking Ensemble...")
        stacking_clf = StackingClassifier(
            estimators=base_models,
            final_estimator=LogisticRegression(random_state=42),
            cv=5,
            n_jobs=-1
        )
        stacking_clf.fit(X_train, y_train)
        
        val_pred = stacking_clf.predict(X_val)
        val_acc = accuracy_score(y_val, val_pred)
        
        ensemble_models['stacking_ensemble'] = stacking_clf
        print(f"✅ Stacking Ensemble - Val Acc: {val_acc:.4f}")
        
        # Save ensemble models
        for name, model in ensemble_models.items():
            model_path = self.models_dir / f"{name}_advanced.pkl"
            joblib.dump(model, model_path)
        
        return ensemble_models
    
    def evaluate_models(self, models, X_test, y_test):
        """Comprehensive model evaluation"""
        print("\n📊 Evaluating all models...")
        
        evaluation_results = {}
        
        for name, model in models.items():
            print(f"\n🔍 Evaluating {name}...")
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            # Classification report
            class_report = classification_report(y_test, y_pred, output_dict=True)
            
            evaluation_results[name] = {
                'accuracy': accuracy,
                'classification_report': class_report,
                'predictions': y_pred.tolist(),
                'probabilities': y_pred_proba.tolist() if y_pred_proba is not None else None
            }
            
            print(f"✅ {name} - Accuracy: {accuracy:.4f}")
        
        return evaluation_results
    
    def run_complete_training(self):
        """Run the complete advanced training pipeline"""
        print("🚀 Starting Advanced GPU-Accelerated Exoplanet Model Training!")
        print("="*70)
        
        start_time = time.time()
        
        try:
            # 1. Load data
            df = self.load_data()
            
            # 2. Feature engineering
            df_features = self.engineer_features(df)
            
            # 3. Prepare features
            X, y = self.prepare_features(df_features)
            
            # 4. Encode labels
            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(y)
            
            # 5. Split data
            from sklearn.model_selection import train_test_split
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
            )
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
            )
            
            # 6. Scale features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            X_test_scaled = self.scaler.transform(X_test)
            
            print(f"\n📊 Data splits:")
            print(f"  Training: {X_train_scaled.shape}")
            print(f"  Validation: {X_val_scaled.shape}")
            print(f"  Testing: {X_test_scaled.shape}")
            
            # 7. Train advanced models
            trained_models, training_results = self.train_advanced_models(
                X_train_scaled, y_train, X_val_scaled, y_val
            )
            
            # 8. Create ensemble models
            if trained_models:
                ensemble_models = self.create_ensemble(
                    trained_models, X_train_scaled, y_train, X_val_scaled, y_val
                )
                all_models = {**trained_models, **ensemble_models}
            else:
                all_models = trained_models
            
            # 9. Final evaluation
            evaluation_results = self.evaluate_models(all_models, X_test_scaled, y_test)
            
            # 10. Save results
            results = {
                'training_results': training_results,
                'evaluation_results': evaluation_results,
                'feature_names': X.columns.tolist(),
                'label_classes': self.label_encoder.classes_.tolist(),
                'training_time': time.time() - start_time
            }
            
            results_path = self.results_dir / "advanced_training_results.json"
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            # Save preprocessing components
            joblib.dump(self.scaler, self.models_dir / "advanced_scaler.pkl")
            joblib.dump(self.label_encoder, self.models_dir / "advanced_label_encoder.pkl")
            
            # Print final results
            print("\n🎉 ADVANCED TRAINING COMPLETE!")
            print("="*50)
            print(f"⏱️  Total training time: {time.time() - start_time:.2f} seconds")
            print(f"🎯 Models trained: {len(all_models)}")
            
            # Best model
            best_model = max(evaluation_results.items(), key=lambda x: x[1]['accuracy'])
            print(f"🏆 Best model: {best_model[0]} (Accuracy: {best_model[1]['accuracy']:.4f})")
            
            return all_models, results
            
        except Exception as e:
            print(f"❌ Training failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None

if __name__ == "__main__":
    # Initialize and run advanced training
    trainer = AdvancedGPUExoplanetTrainer()
    models, results = trainer.run_complete_training()