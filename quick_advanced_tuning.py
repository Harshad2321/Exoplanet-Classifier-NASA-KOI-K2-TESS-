#!/usr/bin/env python3
"""
🚀 Quick Advanced Model Tuning for Exoplanet Classification
Faster version for immediate results with key optimizations
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

import xgboost as xgb
import lightgbm as lgb

import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import joblib
import os
import time

class QuickAdvancedTuner:
    """Quick advanced tuning with essential optimizations"""
    
    def __init__(self, data_path="data/processed", random_state=42):
        self.data_path = data_path
        self.random_state = random_state
        
        # Create results directory
        os.makedirs("models/quick_tuned", exist_ok=True)
        os.makedirs("results/quick_tuning", exist_ok=True)
        
        print("⚡ Quick Advanced Exoplanet Model Tuning")
        print("=" * 50)
        
    def load_and_enhance_features(self):
        """Load data and apply key feature engineering"""
        print("📥 Loading and enhancing features...")
        
        # Load data
        features = pd.read_csv(f"{self.data_path}/features.csv")
        labels = pd.read_csv(f"{self.data_path}/labels.csv")
        
        print(f"📊 Original: {features.shape[0]} samples, {features.shape[1]} features")
        print(f"🏷️ Classes: {labels['label'].value_counts().to_dict()}")
        
        # Key feature engineering
        enhanced = features.copy()
        
        # Log transforms for skewed features
        for col in ['period', 'radius', 'temperature', 'insolation', 'depth']:
            if col in enhanced.columns:
                enhanced[f'{col}_log'] = np.log1p(enhanced[col])
                
        # Key interaction features
        if all(col in enhanced.columns for col in ['period', 'radius']):
            enhanced['period_radius_ratio'] = enhanced['period'] / (enhanced['radius'] + 1e-8)
            enhanced['orbital_compactness'] = enhanced['period'] * enhanced['radius']
            
        if all(col in enhanced.columns for col in ['temperature', 'insolation']):
            enhanced['stellar_energy'] = enhanced['temperature'] * enhanced['insolation']
            enhanced['habitability_index'] = enhanced['temperature'] / (enhanced['insolation'] + 1e-8)
            
        if all(col in enhanced.columns for col in ['depth', 'radius']):
            enhanced['transit_signal'] = enhanced['depth'] * enhanced['radius']
            
        # Remove infinite values
        enhanced = enhanced.replace([np.inf, -np.inf], np.nan)
        enhanced = enhanced.fillna(enhanced.median())
        
        # Encode labels
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = le.fit_transform(labels['label'])
        self.label_encoder = le
        self.class_names = le.classes_
        
        print(f"🔧 Enhanced: {enhanced.shape[1]} features")
        return enhanced, y
        
    def quick_feature_selection(self, X, y, n_features=25):
        """Quick but effective feature selection"""
        print(f"🎯 Selecting top {n_features} features...")
        
        # Statistical selection
        selector = SelectKBest(score_func=f_classif, k=n_features)
        X_selected = selector.fit_transform(X, y)
        
        self.feature_selector = selector
        return X_selected
        
    def train_optimized_models(self, X_train, y_train, X_val, y_val):
        """Train key models with quick optimization"""
        print("🤖 Training optimized models...")
        
        models = {}
        
        # 1. Optimized Random Forest
        print("🌳 Random Forest...")
        rf = RandomForestClassifier(
            n_estimators=500,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=self.random_state,
            n_jobs=-1
        )
        models['RandomForest_Optimized'] = rf
        
        # 2. Extra Trees
        print("🌲 Extra Trees...")
        et = ExtraTreesClassifier(
            n_estimators=500,
            max_depth=20,
            min_samples_split=3,
            min_samples_leaf=1,
            max_features='sqrt',
            random_state=self.random_state,
            n_jobs=-1
        )
        models['ExtraTrees_Optimized'] = et
        
        # 3. Quick XGBoost tuning
        print("🚀 XGBoost with quick tuning...")
        
        # Test a few parameter combinations quickly
        xgb_configs = [
            {'n_estimators': 300, 'max_depth': 6, 'learning_rate': 0.1},
            {'n_estimators': 500, 'max_depth': 8, 'learning_rate': 0.08},
            {'n_estimators': 400, 'max_depth': 7, 'learning_rate': 0.12}
        ]
        
        best_xgb_score = 0
        best_xgb_config = xgb_configs[0]
        
        for config in xgb_configs:
            xgb_model = xgb.XGBClassifier(**config, random_state=self.random_state, n_jobs=-1)
            xgb_model.fit(X_train, y_train)
            score = accuracy_score(y_val, xgb_model.predict(X_val))
            
            if score > best_xgb_score:
                best_xgb_score = score
                best_xgb_config = config
        
        print(f"   Best XGB config: {best_xgb_config} (Score: {best_xgb_score:.4f})")
        models['XGBoost_Optimized'] = xgb.XGBClassifier(**best_xgb_config, random_state=self.random_state, n_jobs=-1)
        
        # 4. Quick LightGBM tuning
        print("💡 LightGBM with quick tuning...")
        
        lgb_configs = [
            {'n_estimators': 300, 'max_depth': 6, 'learning_rate': 0.1, 'num_leaves': 31},
            {'n_estimators': 500, 'max_depth': 8, 'learning_rate': 0.08, 'num_leaves': 63},
            {'n_estimators': 400, 'max_depth': 7, 'learning_rate': 0.12, 'num_leaves': 50}
        ]
        
        best_lgb_score = 0
        best_lgb_config = lgb_configs[0]
        
        for config in lgb_configs:
            config['verbosity'] = -1
            lgb_model = lgb.LGBMClassifier(**config, random_state=self.random_state, n_jobs=-1)
            lgb_model.fit(X_train, y_train)
            score = accuracy_score(y_val, lgb_model.predict(X_val))
            
            if score > best_lgb_score:
                best_lgb_score = score
                best_lgb_config = config
                
        print(f"   Best LGB config: {best_lgb_config} (Score: {best_lgb_score:.4f})")
        models['LightGBM_Optimized'] = lgb.LGBMClassifier(**best_lgb_config, random_state=self.random_state, n_jobs=-1)
        
        # Fit all models
        for name, model in models.items():
            print(f"   🔧 Fitting {name}...")
            model.fit(X_train, y_train)
            
        return models
        
    def create_quick_ensemble(self, models, X_train, y_train):
        """Create voting ensemble from best models"""
        print("🗳️ Creating voting ensemble...")
        
        # Select top 3 models for ensemble (to avoid overfitting)
        estimators = [(name, model) for name, model in list(models.items())[:3]]
        
        voting_ensemble = VotingClassifier(
            estimators=estimators,
            voting='soft',  # Use probabilities
            n_jobs=-1
        )
        
        voting_ensemble.fit(X_train, y_train)
        
        return {'VotingEnsemble': voting_ensemble}
        
    def evaluate_models(self, models, X_test, y_test):
        """Evaluate all models comprehensively"""
        print("📊 Evaluating models...")
        
        results = {}
        
        for name, model in models.items():
            start_time = time.time()
            
            # Predictions
            y_pred = model.predict(X_test)
            
            # Core metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, y_pred, average='weighted', zero_division=0
            )
            
            training_time = time.time() - start_time
            
            results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'time': training_time
            }
            
            print(f"✅ {name:<25} {accuracy:.4f} accuracy ({training_time:.2f}s)")
            
        return results
        
    def analyze_feature_importance(self, models, feature_names):
        """Analyze and visualize feature importance"""
        print("🔍 Analyzing feature importance...")
        
        importance_data = {}
        
        for name, model in models.items():
            if hasattr(model, 'feature_importances_'):
                importance_data[name] = model.feature_importances_
            elif hasattr(model, 'estimators_'):
                # For voting classifier, average importances
                importances = []
                for estimator_info in model.estimators_:
                    if isinstance(estimator_info, tuple):
                        _, estimator = estimator_info
                    else:
                        estimator = estimator_info
                    if hasattr(estimator, 'feature_importances_'):
                        importances.append(estimator.feature_importances_)
                if importances:
                    importance_data[name] = np.mean(importances, axis=0)
                    
        if importance_data:
            # Create feature importance DataFrame
            importance_df = pd.DataFrame(importance_data, index=feature_names[:len(list(importance_data.values())[0])])
            
            # Save feature importance
            importance_df.to_csv("results/quick_tuning/feature_importance.csv")
            
            # Plot top features
            plt.figure(figsize=(12, 8))
            
            # Average importance across models
            avg_importance = importance_df.mean(axis=1).sort_values(ascending=True)
            
            plt.barh(range(len(avg_importance.tail(15))), avg_importance.tail(15).values)
            plt.yticks(range(len(avg_importance.tail(15))), avg_importance.tail(15).index)
            plt.xlabel('Average Feature Importance')
            plt.title('Top 15 Most Important Features')
            plt.tight_layout()
            plt.savefig('results/quick_tuning/feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print("📊 Feature importance saved to results/quick_tuning/")
            
        return importance_data
        
    def generate_detailed_report(self, results, models, X_test, y_test):
        """Generate comprehensive results report"""
        print("📋 Generating detailed report...")
        
        # Sort results by accuracy
        sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        
        # Best model analysis
        best_model_name, best_metrics = sorted_results[0]
        best_model = models[best_model_name]
        
        # Detailed predictions for best model
        y_pred = best_model.predict(X_test)
        
        # Classification report
        class_report = classification_report(
            y_test, y_pred, 
            target_names=self.class_names,
            output_dict=True
        )
        
        # Confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Create results DataFrame
        df_results = pd.DataFrame(results).T
        df_results = df_results.sort_values('accuracy', ascending=False)
        
        # Save detailed results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_path = f"results/quick_tuning/quick_tuning_results_{timestamp}.csv"
        df_results.to_csv(results_path)
        
        # Save classification report
        report_df = pd.DataFrame(class_report).transpose()
        report_df.to_csv(f"results/quick_tuning/classification_report_{timestamp}.csv")
        
        # Confusion matrix plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title(f'Confusion Matrix - {best_model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'results/quick_tuning/confusion_matrix_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("\n🏆 QUICK ADVANCED TUNING RESULTS")
        print("=" * 70)
        print(f"{'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Time':<8}")
        print("-" * 85)
        
        for name, metrics in df_results.iterrows():
            print(f"{name:<25} {metrics['accuracy']:<12.4f} {metrics['precision']:<12.4f} "
                  f"{metrics['recall']:<12.4f} {metrics['f1_score']:<12.4f} {metrics['time']:<8.2f}s")
        
        print("-" * 85)
        best_accuracy = df_results.loc[best_model_name, 'accuracy']
        print(f"🏅 Best Model: {best_model_name}")
        print(f"🎯 Accuracy: {best_accuracy:.4f}")
        print(f"📊 Improvement over baseline: +{(best_accuracy - 0.701)*100:.2f}%")
        print(f"📁 Results saved to: {results_path}")
        
        # Per-class performance
        print(f"\n📈 Per-class Performance ({best_model_name}):")
        for class_name in self.class_names:
            if class_name in class_report:
                metrics = class_report[class_name]
                print(f"   {class_name}: {metrics['precision']:.3f} precision, "
                      f"{metrics['recall']:.3f} recall, {metrics['f1-score']:.3f} f1")
        
        return df_results
        
    def save_best_models(self, models, results):
        """Save the top performing models"""
        print("💾 Saving best models...")
        
        sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        
        for i, (name, metrics) in enumerate(sorted_results[:2]):  # Save top 2
            model_path = f"models/quick_tuned/model_{i+1}_{name.lower().replace(' ', '_')}.joblib"
            joblib.dump(models[name], model_path)
            print(f"💾 {name}: {metrics['accuracy']:.4f} → {model_path}")
            
        # Save preprocessing components
        preprocessing = {
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'label_encoder': self.label_encoder,
            'class_names': self.class_names
        }
        joblib.dump(preprocessing, "models/quick_tuned/preprocessing.joblib")
        print("💾 Preprocessing pipeline → models/quick_tuned/preprocessing.joblib")
        
    def run_quick_tuning(self):
        """Run the complete quick tuning pipeline"""
        print("\n⚡ Starting Quick Advanced Tuning Pipeline")
        print("=" * 50)
        
        # 1. Load and enhance features
        X, y = self.load_and_enhance_features()
        
        # 2. Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=self.random_state
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=self.random_state
        )
        
        print(f"📊 Data splits - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # 3. Scale features
        print("🔧 Scaling features...")
        self.scaler = RobustScaler()  # Robust to outliers
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 4. Feature selection
        X_train_selected = self.quick_feature_selection(X_train_scaled, y_train)
        X_val_selected = self.feature_selector.transform(X_val_scaled)
        X_test_selected = self.feature_selector.transform(X_test_scaled)
        
        # 5. Train optimized models
        models = self.train_optimized_models(X_train_selected, y_train, X_val_selected, y_val)
        
        # 6. Create ensemble
        ensemble_models = self.create_quick_ensemble(models, X_train_selected, y_train)
        all_models = {**models, **ensemble_models}
        
        # 7. Evaluate models
        results = self.evaluate_models(all_models, X_test_selected, y_test)
        
        # 8. Feature importance analysis
        feature_names = [f"feature_{i}" for i in range(X_train_selected.shape[1])]
        self.analyze_feature_importance(all_models, feature_names)
        
        # 9. Generate detailed report
        df_results = self.generate_detailed_report(results, all_models, X_test_selected, y_test)
        
        # 10. Save best models
        self.save_best_models(all_models, results)
        
        print("\n✅ Quick Advanced Tuning Complete!")
        return df_results

def main():
    """Run quick advanced tuning"""
    tuner = QuickAdvancedTuner()
    results = tuner.run_quick_tuning()
    return results

if __name__ == "__main__":
    results = main()