"""
Enhanced Model Training Pipeline for NASA Space Apps Challenge 2025
"A World Away: Hunting for Exoplanets with AI"

This module implements advanced machine learning models including:
- Gradient Boosting, XGBoost, LightGBM
- Ensemble Voting Classifier
- Advanced hyperparameter tuning with Optuna
- Comprehensive model comparison and evaluation
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb
import lightgbm as lgb
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import optuna
from pathlib import Path
import warnings
import time
from datetime import datetime

warnings.filterwarnings('ignore')

class EnhancedExoplanetTrainer:
    """Advanced machine learning trainer for exoplanet classification"""
    
    def __init__(self, data_dir="data", models_dir="models", results_dir="reports"):
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.results_dir = Path(results_dir)
        
        # Create directories
        self.models_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        (self.results_dir / "figures").mkdir(exist_ok=True)
        
        # Initialize model storage
        self.models = {}
        self.model_scores = {}
        self.best_model = None
        self.best_model_name = None
        
    def load_data(self):
        """Load preprocessed data"""
        try:
            print("📊 Loading preprocessed data...")
            
            # Load features and labels
            features = pd.read_csv(self.data_dir / "processed" / "features.csv")
            labels = pd.read_csv(self.data_dir / "processed" / "labels.csv")
            
            print(f"✅ Data loaded: {len(features):,} samples, {len(features.columns)} features")
            print(f"🎯 Classes: {sorted(labels['label'].unique())}")
            
            return features, labels
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            raise
    
    def prepare_data(self, features, labels, test_size=0.2, val_size=0.2):
        """Prepare train/validation/test splits"""
        X = features.values
        y = labels['label'].values
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"📊 Data splits:")
        print(f"   Training: {len(X_train_scaled):,} samples")
        print(f"   Validation: {len(X_val_scaled):,} samples")
        print(f"   Test: {len(X_test_scaled):,} samples")
        
        return (X_train_scaled, X_val_scaled, X_test_scaled, 
                y_train, y_val, y_test, features.columns.tolist())
    
    def create_models(self):
        """Create advanced model configurations"""
        models = {
            'logistic_regression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'params': {
                    'C': [0.1, 1.0, 10.0, 100.0],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                }
            },
            'random_forest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 1.0]
                }
            },
            'xgboost': {
                'model': xgb.XGBClassifier(random_state=42, eval_metric='mlogloss'),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 1.0],
                    'colsample_bytree': [0.8, 1.0]
                }
            },
            'lightgbm': {
                'model': lgb.LGBMClassifier(random_state=42, verbose=-1),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 1.0],
                    'colsample_bytree': [0.8, 1.0]
                }
            },
            'svm': {
                'model': SVC(random_state=42, probability=True),
                'params': {
                    'C': [0.1, 1.0, 10.0],
                    'kernel': ['rbf', 'linear'],
                    'gamma': ['scale', 'auto']
                }
            }
        }
        
        return models
    
    def optimize_hyperparameters_optuna(self, model_name, model_config, X_train, y_train, X_val, y_val):
        """Advanced hyperparameter optimization using Optuna"""
        
        def objective(trial):
            params = {}
            
            if model_name == 'random_forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 30),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
                }
            elif model_name == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
                }
            elif model_name == 'lightgbm':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
                }
            
            # Create model with trial parameters
            if model_name == 'random_forest':
                model = RandomForestClassifier(random_state=42, **params)
            elif model_name == 'xgboost':
                model = xgb.XGBClassifier(random_state=42, eval_metric='mlogloss', **params)
            elif model_name == 'lightgbm':
                model = lgb.LGBMClassifier(random_state=42, verbose=-1, **params)
            else:
                return 0.0
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='f1_macro')
            return cv_scores.mean()
        
        print(f"🔧 Optimizing {model_name} with Optuna...")
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
        study.optimize(objective, n_trials=50, show_progress_bar=False)
        
        return study.best_params, study.best_value
    
    def train_models(self, X_train, X_val, X_test, y_train, y_val, y_test, feature_names):
        """Train all models with advanced hyperparameter tuning"""
        print("🚀 Starting advanced model training...")
        
        models_config = self.create_models()
        
        for model_name, config in models_config.items():
            print(f"\n{'='*50}")
            print(f"🤖 Training {model_name.upper()}")
            print(f"{'='*50}")
            
            start_time = time.time()
            
            try:
                # Use Optuna for tree-based models
                if model_name in ['random_forest', 'xgboost', 'lightgbm']:
                    best_params, best_score = self.optimize_hyperparameters_optuna(
                        model_name, config, X_train, y_train, X_val, y_val
                    )
                    
                    # Create model with best parameters
                    if model_name == 'random_forest':
                        model = RandomForestClassifier(random_state=42, **best_params)
                    elif model_name == 'xgboost':
                        model = xgb.XGBClassifier(random_state=42, eval_metric='mlogloss', **best_params)
                    elif model_name == 'lightgbm':
                        model = lgb.LGBMClassifier(random_state=42, verbose=-1, **best_params)
                        
                else:
                    # Use GridSearch for other models
                    from sklearn.model_selection import GridSearchCV
                    
                    grid_search = GridSearchCV(
                        config['model'], config['params'],
                        cv=5, scoring='f1_macro', n_jobs=-1
                    )
                    grid_search.fit(X_train, y_train)
                    
                    model = grid_search.best_estimator_
                    best_params = grid_search.best_params_
                    best_score = grid_search.best_score_
                
                # Train final model
                model.fit(X_train, y_train)
                
                # Evaluate model
                train_score = model.score(X_train, y_train)
                val_score = model.score(X_val, y_val)
                test_score = model.score(X_test, y_test)
                
                # Get detailed metrics
                y_pred_test = model.predict(X_test)
                test_report = classification_report(y_test, y_pred_test, output_dict=True)
                
                training_time = time.time() - start_time
                
                # Store model and results
                self.models[model_name] = model
                self.model_scores[model_name] = {
                    'best_params': best_params,
                    'cv_score': best_score,
                    'train_score': train_score,
                    'val_score': val_score,
                    'test_score': test_score,
                    'test_f1_macro': test_report['macro avg']['f1-score'],
                    'test_precision': test_report['macro avg']['precision'],
                    'test_recall': test_report['macro avg']['recall'],
                    'training_time': training_time
                }
                
                print(f"✅ {model_name} completed:")
                print(f"   Best params: {best_params}")
                print(f"   CV F1-score: {best_score:.4f}")
                print(f"   Test accuracy: {test_score:.4f}")
                print(f"   Test F1-score: {test_report['macro avg']['f1-score']:.4f}")
                print(f"   Training time: {training_time:.1f}s")
                
            except Exception as e:
                print(f"❌ Error training {model_name}: {e}")
                continue
        
        # Select best model
        best_model_name = max(self.model_scores, key=lambda x: self.model_scores[x]['test_f1_macro'])
        self.best_model = self.models[best_model_name]
        self.best_model_name = best_model_name
        
        print(f"\n🏆 Best model: {best_model_name.upper()}")
        print(f"   Test F1-score: {self.model_scores[best_model_name]['test_f1_macro']:.4f}")
        
        return self.models, self.model_scores
    
    def create_ensemble(self, X_train, y_train):
        """Create ensemble voting classifier"""
        print("\n🤝 Creating ensemble voting classifier...")
        
        # Select top 3 models for ensemble
        sorted_models = sorted(self.model_scores.items(), 
                              key=lambda x: x[1]['test_f1_macro'], 
                              reverse=True)[:3]
        
        ensemble_models = []
        for model_name, _ in sorted_models:
            ensemble_models.append((model_name, self.models[model_name]))
        
        # Create voting classifier
        ensemble = VotingClassifier(estimators=ensemble_models, voting='soft')
        ensemble.fit(X_train, y_train)
        
        self.models['ensemble'] = ensemble
        print(f"✅ Ensemble created with: {[name for name, _ in ensemble_models]}")
        
        return ensemble
    
    def evaluate_ensemble(self, ensemble, X_test, y_test):
        """Evaluate ensemble model"""
        test_score = ensemble.score(X_test, y_test)
        y_pred = ensemble.predict(X_test)
        test_report = classification_report(y_test, y_pred, output_dict=True)
        
        self.model_scores['ensemble'] = {
            'test_score': test_score,
            'test_f1_macro': test_report['macro avg']['f1-score'],
            'test_precision': test_report['macro avg']['precision'],
            'test_recall': test_report['macro avg']['recall']
        }
        
        print(f"🤝 Ensemble Results:")
        print(f"   Test accuracy: {test_score:.4f}")
        print(f"   Test F1-score: {test_report['macro avg']['f1-score']:.4f}")
        
        return test_report
    
    def create_visualizations(self, X_test, y_test, feature_names):
        """Create comprehensive model comparison visualizations"""
        print("\n📊 Creating visualizations...")
        
        # Model comparison chart
        model_names = list(self.model_scores.keys())
        test_scores = [self.model_scores[name].get('test_f1_macro', 
                      self.model_scores[name].get('test_score', 0)) for name in model_names]
        
        fig = go.Figure(data=[
            go.Bar(x=model_names, y=test_scores,
                   marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#74B9FF'])
        ])
        
        fig.update_layout(
            title='🏆 Model Performance Comparison (F1-Score)',
            xaxis_title='Models',
            yaxis_title='F1-Score',
            template='plotly_white'
        )
        
        fig.write_html(self.results_dir / "figures" / "model_comparison.html")
        fig.show()
        
        # Feature importance for best model
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
            feature_imp_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            fig = px.bar(feature_imp_df.head(10), 
                        x='importance', y='feature', 
                        title=f'🔍 Top 10 Feature Importances ({self.best_model_name})',
                        orientation='h')
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            fig.write_html(self.results_dir / "figures" / "feature_importance.html")
            fig.show()
        
        # Confusion matrix for best model
        y_pred = self.best_model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        fig = px.imshow(cm, 
                       text_auto=True,
                       labels=dict(x="Predicted", y="Actual"),
                       x=sorted(set(y_test)),
                       y=sorted(set(y_test)),
                       title=f'🎯 Confusion Matrix ({self.best_model_name})')
        fig.write_html(self.results_dir / "figures" / "confusion_matrix.html")
        fig.show()
        
        print("✅ Visualizations saved to reports/figures/")
    
    def save_models(self, feature_names):
        """Save all trained models and metadata"""
        print(f"\n💾 Saving models...")
        
        # Save best model
        best_model_path = self.models_dir / f"best_model_{self.best_model_name}.joblib"
        joblib.dump(self.best_model, best_model_path)
        
        # Save scaler
        scaler_path = self.models_dir / "scaler.joblib"
        joblib.dump(self.scaler, scaler_path)
        
        # Save all models
        for name, model in self.models.items():
            model_path = self.models_dir / f"{name}_model.joblib"
            joblib.dump(model, model_path)
        
        # Save metadata
        metadata = {
            'best_model': self.best_model_name,
            'feature_names': feature_names,
            'model_scores': self.model_scores,
            'training_date': datetime.now().isoformat(),
            'n_features': len(feature_names)
        }
        
        metadata_path = self.models_dir / "enhanced_model_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Models saved:")
        print(f"   Best model: {best_model_path}")
        print(f"   Scaler: {scaler_path}")
        print(f"   Metadata: {metadata_path}")
        
        return metadata

def main():
    """Main training pipeline"""
    print("🚀 NASA Space Apps Challenge 2025 - Enhanced Model Training")
    print("=" * 70)
    
    # Initialize trainer
    trainer = EnhancedExoplanetTrainer()
    
    # Load data
    features, labels = trainer.load_data()
    
    # Prepare data splits
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = trainer.prepare_data(features, labels)
    
    # Train models
    models, scores = trainer.train_models(X_train, X_val, X_test, y_train, y_val, y_test, feature_names)
    
    # Create ensemble
    ensemble = trainer.create_ensemble(X_train, y_train)
    trainer.evaluate_ensemble(ensemble, X_test, y_test)
    
    # Update best model if ensemble is better
    if trainer.model_scores['ensemble']['test_f1_macro'] > trainer.model_scores[trainer.best_model_name]['test_f1_macro']:
        trainer.best_model = ensemble
        trainer.best_model_name = 'ensemble'
        print(f"\n🏆 Ensemble is the new best model!")
    
    # Create visualizations
    trainer.create_visualizations(X_test, y_test, feature_names)
    
    # Save models
    metadata = trainer.save_models(feature_names)
    
    # Final summary
    print(f"\n🎉 Training Complete!")
    print(f"   Best Model: {trainer.best_model_name}")
    print(f"   Test F1-Score: {trainer.model_scores[trainer.best_model_name]['test_f1_macro']:.4f}")
    print(f"   Models saved to: {trainer.models_dir}")
    print(f"   Visualizations: {trainer.results_dir}/figures/")

if __name__ == "__main__":
    main()