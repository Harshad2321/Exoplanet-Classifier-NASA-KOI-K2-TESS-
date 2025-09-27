"""
Model Training Module for NASA Exoplanet Classification

This module handles:
- Multiple ML model training (Logistic Regression, Random Forest, XGBoost, LightGBM)
- Hyperparameter tuning using Optuna
- Cross-validation and model evaluation
- Model ensemble and selection
- Model serialization and saving

Author: NASA Space Apps Challenge 2025 Team
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import logging
import joblib
import warnings
from datetime import datetime

# ML libraries
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import xgboost as xgb
import lightgbm as lgb

# Hyperparameter tuning
import optuna
from optuna import Trial

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExoplanetModelTrainer:
    """
    Comprehensive model training pipeline for exoplanet classification
    """
    
    def __init__(self, data_dir: str = "data", models_dir: str = "models"):
        """
        Initialize model trainer
        
        Args:
            data_dir: Directory containing processed data
            models_dir: Directory for saving trained models
        """
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.splits_dir = self.data_dir / "splits"
        self.processed_dir = self.data_dir / "processed"
        
        # Create models directory
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model storage
        self.models = {}
        self.model_scores = {}
        self.best_model = None
        self.feature_names = None
        self.target_mapping = None
        
        # Training data
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
    
    def load_processed_data(self) -> bool:
        """
        Load processed training data
        
        Returns:
            Success status
        """
        try:
            # Load train/test splits
            train_df = pd.read_csv(self.splits_dir / 'train.csv')
            test_df = pd.read_csv(self.splits_dir / 'test.csv')
            
            # Load feature names and target mapping
            self.feature_names = joblib.load(self.processed_dir / 'feature_names.pkl')
            self.target_mapping = joblib.load(self.processed_dir / 'target_mapping.pkl')
            
            # Separate features and targets
            self.X_train = train_df[self.feature_names]
            self.y_train = train_df.drop(columns=self.feature_names).iloc[:, 0]
            
            self.X_test = test_df[self.feature_names]
            self.y_test = test_df.drop(columns=self.feature_names).iloc[:, 0]
            
            logger.info(f"✅ Loaded training data: {len(self.X_train)} train, {len(self.X_test)} test samples")
            logger.info(f"Features: {len(self.feature_names)}")
            logger.info(f"Classes: {list(self.target_mapping.keys())}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load processed data: {e}")
            return False
    
    def train_logistic_regression(self, trial: Trial = None) -> Dict[str, Any]:
        """
        Train Logistic Regression model with optional hyperparameter tuning
        
        Args:
            trial: Optuna trial for hyperparameter tuning
            
        Returns:
            Model performance metrics
        """
        logger.info("Training Logistic Regression...")
        
        if trial:
            # Hyperparameter tuning
            C = trial.suggest_float('lr_C', 1e-4, 1e2, log=True)
            max_iter = trial.suggest_int('lr_max_iter', 1000, 5000)
            solver = trial.suggest_categorical('lr_solver', ['liblinear', 'lbfgs'])
        else:
            # Default parameters
            C = 1.0
            max_iter = 2000
            solver = 'lbfgs'
        
        model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            solver=solver,
            random_state=42,
            multi_class='ovr'
        )
        
        # Train model
        model.fit(self.X_train, self.y_train)
        
        # Evaluate
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)
        
        metrics = self._calculate_metrics(self.y_test, y_pred, y_pred_proba)
        
        if not trial:  # Only store if not hyperparameter tuning
            self.models['logistic_regression'] = model
            self.model_scores['logistic_regression'] = metrics
            logger.info(f"Logistic Regression - Accuracy: {metrics['accuracy']:.4f}")
        
        return metrics
    
    def train_random_forest(self, trial: Trial = None) -> Dict[str, Any]:
        """
        Train Random Forest model with optional hyperparameter tuning
        
        Args:
            trial: Optuna trial for hyperparameter tuning
            
        Returns:
            Model performance metrics
        """
        logger.info("Training Random Forest...")
        
        if trial:
            # Hyperparameter tuning
            n_estimators = trial.suggest_int('rf_n_estimators', 50, 300)
            max_depth = trial.suggest_int('rf_max_depth', 5, 20)
            min_samples_split = trial.suggest_int('rf_min_samples_split', 2, 10)
            min_samples_leaf = trial.suggest_int('rf_min_samples_leaf', 1, 5)
        else:
            # Default parameters
            n_estimators = 100
            max_depth = 10
            min_samples_split = 5
            min_samples_leaf = 2
        
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
            n_jobs=-1
        )
        
        # Train model
        model.fit(self.X_train, self.y_train)
        
        # Evaluate
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)
        
        metrics = self._calculate_metrics(self.y_test, y_pred, y_pred_proba)
        
        if not trial:  # Only store if not hyperparameter tuning
            self.models['random_forest'] = model
            self.model_scores['random_forest'] = metrics
            logger.info(f"Random Forest - Accuracy: {metrics['accuracy']:.4f}")
        
        return metrics
    
    def train_xgboost(self, trial: Trial = None) -> Dict[str, Any]:
        """
        Train XGBoost model with optional hyperparameter tuning
        
        Args:
            trial: Optuna trial for hyperparameter tuning
            
        Returns:
            Model performance metrics
        """
        logger.info("Training XGBoost...")
        
        if trial:
            # Hyperparameter tuning
            n_estimators = trial.suggest_int('xgb_n_estimators', 50, 300)
            max_depth = trial.suggest_int('xgb_max_depth', 3, 10)
            learning_rate = trial.suggest_float('xgb_learning_rate', 0.01, 0.3)
            subsample = trial.suggest_float('xgb_subsample', 0.6, 1.0)
            colsample_bytree = trial.suggest_float('xgb_colsample_bytree', 0.6, 1.0)
        else:
            # Default parameters
            n_estimators = 100
            max_depth = 6
            learning_rate = 0.1
            subsample = 0.8
            colsample_bytree = 0.8
        
        model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=42,
            eval_metric='logloss',
            verbosity=0
        )
        
        # Train model
        model.fit(self.X_train, self.y_train)
        
        # Evaluate
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)
        
        metrics = self._calculate_metrics(self.y_test, y_pred, y_pred_proba)
        
        if not trial:  # Only store if not hyperparameter tuning
            self.models['xgboost'] = model
            self.model_scores['xgboost'] = metrics
            logger.info(f"XGBoost - Accuracy: {metrics['accuracy']:.4f}")
        
        return metrics
    
    def train_lightgbm(self, trial: Trial = None) -> Dict[str, Any]:
        """
        Train LightGBM model with optional hyperparameter tuning
        
        Args:
            trial: Optuna trial for hyperparameter tuning
            
        Returns:
            Model performance metrics
        """
        logger.info("Training LightGBM...")
        
        if trial:
            # Hyperparameter tuning
            n_estimators = trial.suggest_int('lgb_n_estimators', 50, 300)
            max_depth = trial.suggest_int('lgb_max_depth', 3, 10)
            learning_rate = trial.suggest_float('lgb_learning_rate', 0.01, 0.3)
            num_leaves = trial.suggest_int('lgb_num_leaves', 10, 100)
            subsample = trial.suggest_float('lgb_subsample', 0.6, 1.0)
        else:
            # Default parameters
            n_estimators = 100
            max_depth = 6
            learning_rate = 0.1
            num_leaves = 31
            subsample = 0.8
        
        model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            subsample=subsample,
            random_state=42,
            verbosity=-1
        )
        
        # Train model
        model.fit(self.X_train, self.y_train)
        
        # Evaluate
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)
        
        metrics = self._calculate_metrics(self.y_test, y_pred, y_pred_proba)
        
        if not trial:  # Only store if not hyperparameter tuning
            self.models['lightgbm'] = model
            self.model_scores['lightgbm'] = metrics
            logger.info(f"LightGBM - Accuracy: {metrics['accuracy']:.4f}")
        
        return metrics
    
    def train_all_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Train all baseline models
        
        Returns:
            Dictionary of model performance metrics
        """
        logger.info("🚀 Training all baseline models...")
        
        # Train individual models
        self.train_logistic_regression()
        self.train_random_forest()
        self.train_xgboost()
        self.train_lightgbm()
        
        # Create ensemble model
        self._create_ensemble_model()
        
        logger.info("✅ All models trained successfully!")
        
        return self.model_scores
    
    def _create_ensemble_model(self):
        """Create ensemble model from trained individual models"""
        logger.info("Creating ensemble model...")
        
        if len(self.models) < 2:
            logger.warning("Need at least 2 models for ensemble")
            return
        
        # Create voting classifier from trained models
        estimators = [(name, model) for name, model in self.models.items()]
        
        ensemble = VotingClassifier(
            estimators=estimators,
            voting='soft'  # Use probability averaging
        )
        
        # Fit ensemble (this just sets up the ensemble, individual models are already trained)
        ensemble.fit(self.X_train, self.y_train)
        
        # Evaluate ensemble
        y_pred = ensemble.predict(self.X_test)
        y_pred_proba = ensemble.predict_proba(self.X_test)
        
        metrics = self._calculate_metrics(self.y_test, y_pred, y_pred_proba)
        
        self.models['ensemble'] = ensemble
        self.model_scores['ensemble'] = metrics
        
        logger.info(f"Ensemble - Accuracy: {metrics['accuracy']:.4f}")
    
    def hyperparameter_tuning(self, model_name: str, n_trials: int = 100) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning for a specific model
        
        Args:
            model_name: Name of model to tune ('logistic_regression', 'random_forest', etc.)
            n_trials: Number of optimization trials
            
        Returns:
            Best hyperparameters and metrics
        """
        logger.info(f"Starting hyperparameter tuning for {model_name}...")
        
        def objective(trial):
            if model_name == 'logistic_regression':
                metrics = self.train_logistic_regression(trial)
            elif model_name == 'random_forest':
                metrics = self.train_random_forest(trial)
            elif model_name == 'xgboost':
                metrics = self.train_xgboost(trial)
            elif model_name == 'lightgbm':
                metrics = self.train_lightgbm(trial)
            else:
                raise ValueError(f"Unknown model: {model_name}")
            
            return metrics['f1_weighted']  # Optimize for F1-score
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            study_name=f"{model_name}_tuning"
        )
        
        # Optimize
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        # Get best parameters and retrain with them
        best_params = study.best_params
        logger.info(f"Best parameters for {model_name}: {best_params}")
        
        # Retrain model with best parameters
        best_trial = study.best_trial
        best_metrics = self._retrain_with_best_params(model_name, best_params)
        
        return {
            'best_params': best_params,
            'best_metrics': best_metrics,
            'best_value': study.best_value,
            'study': study
        }
    
    def _retrain_with_best_params(self, model_name: str, best_params: Dict) -> Dict[str, Any]:
        """Retrain model with best hyperparameters"""
        
        if model_name == 'logistic_regression':
            model = LogisticRegression(
                C=best_params['lr_C'],
                max_iter=best_params['lr_max_iter'],
                solver=best_params['lr_solver'],
                random_state=42,
                multi_class='ovr'
            )
        elif model_name == 'random_forest':
            model = RandomForestClassifier(
                n_estimators=best_params['rf_n_estimators'],
                max_depth=best_params['rf_max_depth'],
                min_samples_split=best_params['rf_min_samples_split'],
                min_samples_leaf=best_params['rf_min_samples_leaf'],
                random_state=42,
                n_jobs=-1
            )
        elif model_name == 'xgboost':
            model = xgb.XGBClassifier(
                n_estimators=best_params['xgb_n_estimators'],
                max_depth=best_params['xgb_max_depth'],
                learning_rate=best_params['xgb_learning_rate'],
                subsample=best_params['xgb_subsample'],
                colsample_bytree=best_params['xgb_colsample_bytree'],
                random_state=42,
                eval_metric='logloss',
                verbosity=0
            )
        elif model_name == 'lightgbm':
            model = lgb.LGBMClassifier(
                n_estimators=best_params['lgb_n_estimators'],
                max_depth=best_params['lgb_max_depth'],
                learning_rate=best_params['lgb_learning_rate'],
                num_leaves=best_params['lgb_num_leaves'],
                subsample=best_params['lgb_subsample'],
                random_state=42,
                verbosity=-1
            )
        
        # Train and evaluate
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)
        
        metrics = self._calculate_metrics(self.y_test, y_pred, y_pred_proba)
        
        # Update stored model
        self.models[model_name] = model
        self.model_scores[model_name] = metrics
        
        return metrics
    
    def _calculate_metrics(self, y_true, y_pred, y_pred_proba) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics"""
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        # Add ROC-AUC if multiclass
        try:
            if len(np.unique(y_true)) > 2:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
            else:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
        except:
            metrics['roc_auc'] = 0.0
        
        return metrics
    
    def cross_validate_models(self, cv_folds: int = 5) -> Dict[str, Dict[str, float]]:
        """
        Perform cross-validation for all trained models
        
        Args:
            cv_folds: Number of cross-validation folds
            
        Returns:
            Cross-validation scores for each model
        """
        logger.info(f"Performing {cv_folds}-fold cross-validation...")
        
        cv_scores = {}
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Combine train and test for CV (excluding ensemble which needs pre-fitted models)
        X_all = pd.concat([self.X_train, self.X_test])
        y_all = pd.concat([self.y_train, self.y_test])
        
        for name, model in self.models.items():
            if name == 'ensemble':  # Skip ensemble for CV
                continue
                
            logger.info(f"Cross-validating {name}...")
            
            # Calculate CV scores
            cv_accuracy = cross_val_score(model, X_all, y_all, cv=cv, scoring='accuracy')
            cv_f1 = cross_val_score(model, X_all, y_all, cv=cv, scoring='f1_weighted')
            cv_precision = cross_val_score(model, X_all, y_all, cv=cv, scoring='precision_weighted')
            cv_recall = cross_val_score(model, X_all, y_all, cv=cv, scoring='recall_weighted')
            
            cv_scores[name] = {
                'accuracy_mean': cv_accuracy.mean(),
                'accuracy_std': cv_accuracy.std(),
                'f1_mean': cv_f1.mean(),
                'f1_std': cv_f1.std(),
                'precision_mean': cv_precision.mean(),
                'precision_std': cv_precision.std(),
                'recall_mean': cv_recall.mean(),
                'recall_std': cv_recall.std()
            }
        
        return cv_scores
    
    def select_best_model(self, metric: str = 'f1_weighted') -> str:
        """
        Select best model based on specified metric
        
        Args:
            metric: Metric to use for selection
            
        Returns:
            Name of best model
        """
        if not self.model_scores:
            logger.error("No models trained yet")
            return None
        
        best_score = -1
        best_model_name = None
        
        for name, scores in self.model_scores.items():
            if scores[metric] > best_score:
                best_score = scores[metric]
                best_model_name = name
        
        self.best_model = self.models[best_model_name]
        logger.info(f"Best model: {best_model_name} ({metric}={best_score:.4f})")
        
        return best_model_name
    
    def save_models(self, save_all: bool = False):
        """
        Save trained models
        
        Args:
            save_all: Whether to save all models or just the best one
        """
        logger.info("Saving trained models...")
        
        if save_all:
            # Save all models
            for name, model in self.models.items():
                model_path = self.models_dir / f"{name}_model.pkl"
                joblib.dump(model, model_path)
                logger.info(f"Saved {name} to {model_path}")
        else:
            # Save best model only
            if self.best_model is None:
                self.select_best_model()
            
            if self.best_model:
                best_name = self.select_best_model()
                model_path = self.models_dir / "best_model.pkl"
                joblib.dump(self.best_model, model_path)
                logger.info(f"Saved best model ({best_name}) to {model_path}")
        
        # Save model scores and metadata
        scores_path = self.models_dir / "model_scores.pkl"
        joblib.dump(self.model_scores, scores_path)
        
        metadata = {
            'feature_names': self.feature_names,
            'target_mapping': self.target_mapping,
            'timestamp': datetime.now().isoformat(),
            'best_model': self.select_best_model() if self.best_model else None
        }
        
        metadata_path = self.models_dir / "model_metadata.pkl"
        joblib.dump(metadata, metadata_path)
        
        logger.info("✅ Models saved successfully!")
    
    def generate_model_comparison_report(self) -> pd.DataFrame:
        """
        Generate comprehensive model comparison report
        
        Returns:
            DataFrame with model comparison metrics
        """
        if not self.model_scores:
            logger.error("No models trained yet")
            return pd.DataFrame()
        
        # Create comparison DataFrame
        comparison_data = []
        
        for model_name, scores in self.model_scores.items():
            comparison_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'Accuracy': f"{scores['accuracy']:.4f}",
                'Precision (Weighted)': f"{scores['precision_weighted']:.4f}",
                'Recall (Weighted)': f"{scores['recall_weighted']:.4f}",
                'F1-Score (Weighted)': f"{scores['f1_weighted']:.4f}",
                'ROC-AUC': f"{scores['roc_auc']:.4f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Sort by F1-Score
        comparison_df['F1_numeric'] = comparison_df['F1-Score (Weighted)'].astype(float)
        comparison_df = comparison_df.sort_values('F1_numeric', ascending=False)
        comparison_df = comparison_df.drop('F1_numeric', axis=1)
        
        return comparison_df


def main():
    """
    Main function to demonstrate training functionality
    """
    print("🤖 NASA Exoplanet Model Training")
    print("=" * 50)
    
    trainer = ExoplanetModelTrainer()
    
    # Load processed data
    if not trainer.load_processed_data():
        print("❌ Failed to load processed data. Run preprocessing first.")
        return
    
    # Train all models
    print("\n🚀 Training models...")
    scores = trainer.train_all_models()
    
    # Generate comparison report
    print("\n📊 Model Comparison:")
    comparison_df = trainer.generate_model_comparison_report()
    print(comparison_df.to_string(index=False))
    
    # Select and save best model
    best_model = trainer.select_best_model()
    trainer.save_models()
    
    print(f"\n✅ Training complete! Best model: {best_model}")


if __name__ == "__main__":
    main()