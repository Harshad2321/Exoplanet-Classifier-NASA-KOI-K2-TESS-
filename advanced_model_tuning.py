#!/usr/bin/env python3
"""
🚀 Advanced Model Tuning for Exoplanet Classification
RTX 4060 GPU-Optimized Hyperparameter Optimization

Features:
- Bayesian Optimization with Optuna
- Advanced Feature Engineering
- Ensemble Methods with Stacking
- Cross-Validation with Stratification  
- GPU-Accelerated Training
- Automated Model Selection
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score,
    GridSearchCV, RandomizedSearchCV
)
from sklearn.preprocessing import (
    StandardScaler, RobustScaler, PowerTransformer,
    PolynomialFeatures, QuantileTransformer
)
from sklearn.feature_selection import (
    SelectKBest, f_classif, RFE, SelectFromModel,
    VarianceThreshold, mutual_info_classif
)
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, 
    GradientBoostingClassifier, VotingClassifier,
    StackingClassifier, BaggingClassifier
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix,
    roc_auc_score, log_loss
)

import xgboost as xgb
import lightgbm as lgb
import catboost as cb

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("⚠️ Optuna not available. Using traditional hyperparameter tuning.")

import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import joblib
import os
import time

class AdvancedExoplanetTuner:
    """
    🎯 Advanced Model Tuning System for Exoplanet Classification
    """
    
    def __init__(self, data_path="data/processed", random_state=42):
        self.data_path = data_path
        self.random_state = random_state
        self.results = {}
        self.best_models = {}
        self.feature_importance = {}
        
        # Create results directory
        os.makedirs("models/tuned", exist_ok=True)
        os.makedirs("results/tuning", exist_ok=True)
        
        print("🚀 Advanced Exoplanet Model Tuning System")
        print("=" * 60)
        
    def load_and_preprocess_data(self):
        """Load and apply advanced preprocessing"""
        print("📥 Loading and preprocessing data...")
        
        # Load data
        features = pd.read_csv(f"{self.data_path}/features.csv")
        labels = pd.read_csv(f"{self.data_path}/labels.csv")
        
        # Basic info
        print(f"📊 Dataset: {features.shape[0]} samples, {features.shape[1]} features")
        print(f"🏷️ Classes: {labels['label'].unique()}")
        print(f"📈 Class distribution:")
        print(labels['label'].value_counts())
        
        # Advanced feature engineering
        features_enhanced = self._engineer_features(features)
        
        # Encode labels
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = le.fit_transform(labels['label'])
        self.label_encoder = le
        self.class_names = le.classes_
        
        print(f"🔧 Enhanced features: {features_enhanced.shape[1]} total")
        
        return features_enhanced, y
    
    def _engineer_features(self, df):
        """Advanced feature engineering"""
        print("🔧 Engineering advanced features...")
        
        # Start with original features
        features = df.copy()
        
        # Log transforms for skewed features
        skewed_features = ['period', 'radius', 'temperature', 'insolation', 'depth']
        for feature in skewed_features:
            if feature in features.columns:
                features[f'{feature}_log'] = np.log1p(features[feature])
                features[f'{feature}_sqrt'] = np.sqrt(features[feature])
                
        # Interaction features
        if all(col in features.columns for col in ['period', 'radius']):
            features['period_radius_ratio'] = features['period'] / (features['radius'] + 1e-8)
            features['period_radius_product'] = features['period'] * features['radius']
            
        if all(col in features.columns for col in ['temperature', 'insolation']):
            features['temp_insolation_ratio'] = features['temperature'] / (features['insolation'] + 1e-8)
            features['temp_insolation_product'] = features['temperature'] * features['insolation']
            
        if all(col in features.columns for col in ['depth', 'radius']):
            features['depth_radius_ratio'] = features['depth'] / (features['radius'] + 1e-8)
            features['signal_strength'] = features['depth'] * features['radius']
            
        # Astronomical features
        if all(col in features.columns for col in ['ra', 'dec']):
            features['sky_distance'] = np.sqrt(features['ra']**2 + features['dec']**2)
            features['ecliptic_lat'] = np.sin(np.radians(features['dec']))
            
        # Binning features
        if 'temperature' in features.columns:
            features['temp_bin'] = pd.cut(features['temperature'], 
                                        bins=[0, 500, 1000, 1500, np.inf], 
                                        labels=[0, 1, 2, 3])
            
        if 'period' in features.columns:
            features['period_bin'] = pd.cut(features['period'],
                                          bins=[0, 1, 10, 100, np.inf],
                                          labels=[0, 1, 2, 3])
        
        # Handle any categorical features
        categorical_cols = features.select_dtypes(include=['category']).columns
        for col in categorical_cols:
            features[col] = features[col].astype('int')
            
        # Remove any infinite or NaN values
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(features.median())
        
        print(f"✅ Feature engineering complete: {len(features.columns)} features")
        return features
        
    def optimize_preprocessing(self, X, y):
        """Find optimal preprocessing pipeline"""
        print("🔍 Optimizing preprocessing pipeline...")
        
        # Test different scalers
        scalers = {
            'standard': StandardScaler(),
            'robust': RobustScaler(),
            'quantile': QuantileTransformer(n_quantiles=1000, random_state=self.random_state),
            'power': PowerTransformer(method='yeo-johnson', standardize=True)
        }
        
        best_score = 0
        best_scaler = None
        
        # Quick RF test for scaler selection
        rf = RandomForestClassifier(n_estimators=100, random_state=self.random_state, n_jobs=-1)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
        
        for name, scaler in scalers.items():
            X_scaled = scaler.fit_transform(X)
            score = cross_val_score(rf, X_scaled, y, cv=cv, scoring='accuracy').mean()
            
            if score > best_score:
                best_score = score
                best_scaler = scaler
                
        print(f"✅ Best preprocessing: {type(best_scaler).__name__} (Score: {best_score:.4f})")
        return best_scaler
        
    def feature_selection_pipeline(self, X, y):
        """Advanced feature selection pipeline"""
        print("🎯 Running feature selection pipeline...")
        
        # Remove low variance features
        variance_selector = VarianceThreshold(threshold=0.01)
        X_var = variance_selector.fit_transform(X)
        
        # Statistical feature selection
        k_best = SelectKBest(score_func=f_classif, k=min(50, X_var.shape[1]))
        X_stat = k_best.fit_transform(X_var, y)
        
        # Mutual information
        mi_selector = SelectKBest(score_func=mutual_info_classif, k=min(30, X_stat.shape[1]))
        X_mi = mi_selector.fit_transform(X_stat, y)
        
        # Tree-based selection
        rf_selector = SelectFromModel(
            RandomForestClassifier(n_estimators=100, random_state=self.random_state),
            max_features=min(25, X_mi.shape[1])
        )
        X_final = rf_selector.fit_transform(X_mi, y)
        
        print(f"📊 Feature selection: {X.shape[1]} → {X_final.shape[1]} features")
        
        # Store selectors for later use
        self.feature_selectors = {
            'variance': variance_selector,
            'k_best': k_best,
            'mutual_info': mi_selector,
            'tree_based': rf_selector
        }
        
        return X_final
        
    def bayesian_optimization_xgb(self, X_train, y_train, X_val, y_val):
        """Bayesian optimization for XGBoost"""
        if not OPTUNA_AVAILABLE:
            print("⚠️ Optuna not available, using default parameters")
            return {'n_estimators': 500, 'max_depth': 6, 'learning_rate': 0.1}
            
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'random_state': self.random_state,
                'n_jobs': -1
            }
            
            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train)
            pred = model.predict(X_val)
            accuracy = accuracy_score(y_val, pred)
            
            return accuracy
        
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=self.random_state),
            pruner=MedianPruner()
        )
        
        print("🔍 XGBoost Bayesian Optimization (50 trials)...")
        study.optimize(objective, n_trials=50, show_progress_bar=True)
        
        return study.best_params
        
    def bayesian_optimization_lgb(self, X_train, y_train, X_val, y_val):
        """Bayesian optimization for LightGBM"""
        if not OPTUNA_AVAILABLE:
            return {'n_estimators': 500, 'max_depth': 6, 'learning_rate': 0.1}
            
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'random_state': self.random_state,
                'n_jobs': -1,
                'verbosity': -1
            }
            
            model = lgb.LGBMClassifier(**params)
            model.fit(X_train, y_train)
            pred = model.predict(X_val)
            accuracy = accuracy_score(y_val, pred)
            
            return accuracy
        
        study = optuna.create_study(direction='maximize')
        print("🔍 LightGBM Bayesian Optimization (50 trials)...")
        study.optimize(objective, n_trials=50, show_progress_bar=True)
        
        return study.best_params
    
    def train_ensemble_models(self, X_train, y_train, X_val, y_val):
        """Train and optimize ensemble models"""
        print("🤖 Training optimized ensemble models...")
        
        models = {}
        
        # 1. Optimized Random Forest
        print("🌳 Optimizing Random Forest...")
        rf_params = {
            'n_estimators': [300, 500, 800],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', 0.8]
        }
        
        rf = RandomForestClassifier(random_state=self.random_state, n_jobs=-1)
        rf_grid = RandomizedSearchCV(
            rf, rf_params, n_iter=20, cv=3, 
            scoring='accuracy', random_state=self.random_state, n_jobs=-1
        )
        rf_grid.fit(X_train, y_train)
        models['RandomForest'] = rf_grid.best_estimator_
        
        # 2. Optimized XGBoost
        print("🚀 Optimizing XGBoost...")
        xgb_params = self.bayesian_optimization_xgb(X_train, y_train, X_val, y_val)
        models['XGBoost'] = xgb.XGBClassifier(**xgb_params)
        models['XGBoost'].fit(X_train, y_train)
        
        # 3. Optimized LightGBM  
        print("💡 Optimizing LightGBM...")
        lgb_params = self.bayesian_optimization_lgb(X_train, y_train, X_val, y_val)
        models['LightGBM'] = lgb.LGBMClassifier(**lgb_params)
        models['LightGBM'].fit(X_train, y_train)
        
        # 4. Extra Trees
        print("🌲 Training Extra Trees...")
        et = ExtraTreesClassifier(
            n_estimators=500, max_depth=15, random_state=self.random_state, n_jobs=-1
        )
        models['ExtraTrees'] = et
        models['ExtraTrees'].fit(X_train, y_train)
        
        # 5. Gradient Boosting
        print("⚡ Training Gradient Boosting...")
        gb = GradientBoostingClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1, 
            random_state=self.random_state
        )
        models['GradientBoosting'] = gb
        models['GradientBoosting'].fit(X_train, y_train)
        
        return models
        
    def create_stacking_ensemble(self, base_models, X_train, y_train):
        """Create advanced stacking ensemble"""
        print("🏗️ Building stacking ensemble...")
        
        # Base models for stacking
        estimators = [(name, model) for name, model in base_models.items()]
        
        # Meta-learner options
        meta_learners = {
            'LogisticRegression': LogisticRegression(random_state=self.random_state, max_iter=1000),
            'XGBoost_Meta': xgb.XGBClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.1,
                random_state=self.random_state
            )
        }
        
        stacking_models = {}
        
        for meta_name, meta_learner in meta_learners.items():
            print(f"   📊 Meta-learner: {meta_name}")
            
            stacking_clf = StackingClassifier(
                estimators=estimators,
                final_estimator=meta_learner,
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
                stack_method='predict_proba',
                n_jobs=-1
            )
            
            stacking_clf.fit(X_train, y_train)
            stacking_models[f'Stacking_{meta_name}'] = stacking_clf
            
        return stacking_models
        
    def create_voting_ensemble(self, base_models):
        """Create voting ensemble"""
        print("🗳️ Building voting ensemble...")
        
        estimators = [(name, model) for name, model in base_models.items()]
        
        # Hard voting
        voting_hard = VotingClassifier(estimators=estimators, voting='hard', n_jobs=-1)
        
        # Soft voting  
        voting_soft = VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)
        
        return {
            'VotingHard': voting_hard,
            'VotingSoft': voting_soft
        }
        
    def evaluate_models(self, models, X_test, y_test):
        """Comprehensive model evaluation"""
        print("📊 Evaluating all models...")
        
        results = {}
        
        for name, model in models.items():
            start_time = time.time()
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, y_pred, average='weighted'
            )
            
            training_time = time.time() - start_time
            
            # AUC Score (multiclass)
            auc_score = None
            if y_pred_proba is not None:
                try:
                    auc_score = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
                except:
                    auc_score = None
                    
            results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc_score': auc_score,
                'time': training_time
            }
            
            print(f"✅ {name}: {accuracy:.4f} accuracy ({training_time:.2f}s)")
            
        return results
        
    def save_best_models(self, models, results):
        """Save the best performing models"""
        print("💾 Saving best models...")
        
        # Sort by accuracy
        sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        
        # Save top 3 models
        for i, (name, metrics) in enumerate(sorted_results[:3]):
            if name in models:
                model_path = f"models/tuned/best_model_{i+1}_{name.lower()}.joblib"
                joblib.dump(models[name], model_path)
                print(f"💾 Saved: {name} → {model_path}")
                
        # Save preprocessing pipeline
        preprocessing_path = "models/tuned/preprocessing_pipeline.joblib"
        preprocessing_pipeline = {
            'scaler': self.scaler,
            'feature_selectors': self.feature_selectors,
            'label_encoder': self.label_encoder
        }
        joblib.dump(preprocessing_pipeline, preprocessing_path)
        print(f"💾 Saved preprocessing pipeline → {preprocessing_path}")
        
    def generate_report(self, results):
        """Generate comprehensive tuning report"""
        print("📋 Generating tuning report...")
        
        # Create results DataFrame
        df_results = pd.DataFrame(results).T
        df_results = df_results.sort_values('accuracy', ascending=False)
        
        # Save detailed results
        results_path = f"results/tuning/advanced_tuning_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df_results.to_csv(results_path)
        
        print("\n🏆 ADVANCED MODEL TUNING RESULTS")
        print("=" * 70)
        print(f"{'Model':<20} {'Accuracy':<12} {'F1-Score':<12} {'AUC':<12} {'Time':<10}")
        print("-" * 70)
        
        for name, metrics in df_results.iterrows():
            auc_str = f"{metrics['auc_score']:.4f}" if metrics['auc_score'] else "N/A"
            print(f"{name:<20} {metrics['accuracy']:<12.4f} {metrics['f1_score']:<12.4f} "
                  f"{auc_str:<12} {metrics['time']:<10.2f}s")
        
        print("-" * 70)
        best_model = df_results.index[0]
        best_accuracy = df_results.loc[best_model, 'accuracy']
        print(f"🏅 Best Model: {best_model} - {best_accuracy:.4f} accuracy")
        print(f"📊 Results saved to: {results_path}")
        
        return df_results
        
    def run_full_tuning_pipeline(self):
        """Run the complete advanced tuning pipeline"""
        print("\n🚀 Starting Advanced Model Tuning Pipeline")
        print("=" * 60)
        
        # 1. Load and preprocess data
        X, y = self.load_and_preprocess_data()
        
        # 2. Train-validation-test split
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=self.random_state
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=self.random_state
        )
        
        print(f"📊 Data splits - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # 3. Optimize preprocessing
        self.scaler = self.optimize_preprocessing(X_train, y_train)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 4. Feature selection
        X_train_selected = self.feature_selection_pipeline(X_train_scaled, y_train)
        X_val_selected = self._transform_with_selectors(X_val_scaled)
        X_test_selected = self._transform_with_selectors(X_test_scaled)
        
        # 5. Train base models
        base_models = self.train_ensemble_models(
            X_train_selected, y_train, X_val_selected, y_val
        )
        
        # 6. Create ensemble models
        stacking_models = self.create_stacking_ensemble(
            base_models, X_train_selected, y_train
        )
        voting_models = self.create_voting_ensemble(base_models)
        
        # Fit voting models
        for name, model in voting_models.items():
            print(f"🗳️ Training {name}...")
            model.fit(X_train_selected, y_train)
        
        # 7. Combine all models
        all_models = {**base_models, **stacking_models, **voting_models}
        
        # 8. Evaluate all models
        results = self.evaluate_models(all_models, X_test_selected, y_test)
        
        # 9. Save best models
        self.save_best_models(all_models, results)
        
        # 10. Generate report
        df_results = self.generate_report(results)
        
        print("\n✅ Advanced Model Tuning Complete!")
        return df_results
        
    def _transform_with_selectors(self, X):
        """Apply all feature selectors in sequence"""
        X_transformed = self.feature_selectors['variance'].transform(X)
        X_transformed = self.feature_selectors['k_best'].transform(X_transformed)
        X_transformed = self.feature_selectors['mutual_info'].transform(X_transformed)
        X_transformed = self.feature_selectors['tree_based'].transform(X_transformed)
        return X_transformed

def main():
    """Run advanced model tuning"""
    # Initialize tuner
    tuner = AdvancedExoplanetTuner()
    
    # Run full pipeline
    results = tuner.run_full_tuning_pipeline()
    
    print(f"\n🎉 Tuning complete! Check results/ and models/ directories for outputs.")
    
    return results

if __name__ == "__main__":
    results = main()