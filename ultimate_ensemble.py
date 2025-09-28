#!/usr/bin/env python3
"""
🏆 Ultimate Exoplanet Classifier Ensemble
Combines the best of everything: Traditional ML + Neural Networks + Advanced Tuning
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

class UltimateExoplanetEnsemble:
    """Ultimate ensemble combining all our best models"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        
        # Create results directory
        os.makedirs("models/ultimate_ensemble", exist_ok=True)
        os.makedirs("results/ultimate_ensemble", exist_ok=True)
        
        print("🏆 Ultimate Exoplanet Classifier Ensemble")
        print("=" * 50)
        
    def load_best_models(self):
        """Load the best models from previous training sessions"""
        print("📥 Loading best models from previous training...")
        
        model_paths = [
            ("models/quick_tuned/model_1_extratrees_optimized.joblib", "ExtraTrees"),
            ("models/quick_tuned/model_2_votingensemble.joblib", "VotingEnsemble"),
        ]
        
        loaded_models = {}
        
        for path, name in model_paths:
            try:
                if os.path.exists(path):
                    model = joblib.load(path)
                    loaded_models[name] = model
                    print(f"✅ Loaded {name}")
                else:
                    print(f"⚠️ {path} not found")
            except Exception as e:
                print(f"❌ Failed to load {name}: {str(e)}")
                
        # Load preprocessing
        try:
            preprocessing = joblib.load("models/quick_tuned/preprocessing.joblib")
            self.scaler = preprocessing['scaler']
            self.feature_selector = preprocessing['feature_selector']
            self.label_encoder = preprocessing['label_encoder']
            self.class_names = preprocessing['class_names']
            print("✅ Loaded preprocessing pipeline")
        except Exception as e:
            print(f"⚠️ Preprocessing pipeline not found: {str(e)}")
            return None, None
            
        return loaded_models, preprocessing
        
    def create_hybrid_features(self, features_df):
        """Create the same enhanced features used in training"""
        print("🔧 Creating hybrid features...")
        
        enhanced = features_df.copy()
        
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
        
        return enhanced
        
    def train_additional_models(self, X_train, y_train):
        """Train additional specialized models for the ensemble"""
        print("🤖 Training additional specialized models...")
        
        from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
        from sklearn.svm import SVC
        from sklearn.neighbors import KNeighborsClassifier
        import xgboost as xgb
        import lightgbm as lgb
        
        additional_models = {}
        
        # Ultra-tuned Random Forest
        print("🌳 Ultra-tuned Random Forest...")
        rf_ultra = RandomForestClassifier(
            n_estimators=1000,
            max_depth=20,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='log2',
            bootstrap=True,
            oob_score=True,
            random_state=self.random_state,
            n_jobs=-1
        )
        rf_ultra.fit(X_train, y_train)
        additional_models['RF_Ultra'] = rf_ultra
        
        # Diversified Extra Trees
        print("🌲 Diversified Extra Trees...")
        et_diverse = ExtraTreesClassifier(
            n_estimators=800,
            max_depth=25,
            min_samples_split=3,
            min_samples_leaf=1,
            max_features='sqrt',
            bootstrap=False,
            random_state=self.random_state + 1,  # Different seed for diversity
            n_jobs=-1
        )
        et_diverse.fit(X_train, y_train)
        additional_models['ET_Diverse'] = et_diverse
        
        # High-precision XGBoost
        print("🚀 High-precision XGBoost...")
        xgb_precision = xgb.XGBClassifier(
            n_estimators=600,
            max_depth=9,
            learning_rate=0.05,  # Slower learning
            subsample=0.9,
            colsample_bytree=0.8,
            reg_alpha=1,
            reg_lambda=2,
            random_state=self.random_state,
            n_jobs=-1
        )
        xgb_precision.fit(X_train, y_train)
        additional_models['XGB_Precision'] = xgb_precision
        
        # Conservative LightGBM
        print("💡 Conservative LightGBM...")
        lgb_conservative = lgb.LGBMClassifier(
            n_estimators=700,
            max_depth=10,
            learning_rate=0.06,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.5,
            reg_lambda=1.5,
            num_leaves=100,
            random_state=self.random_state,
            n_jobs=-1,
            verbosity=-1
        )
        lgb_conservative.fit(X_train, y_train)
        additional_models['LGB_Conservative'] = lgb_conservative
        
        return additional_models
        
    def create_meta_ensemble(self, base_models, X_train, y_train):
        """Create a meta-ensemble using stacking"""
        print("🏗️ Creating meta-ensemble with stacking...")
        
        from sklearn.ensemble import StackingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold
        
        # Prepare base estimators
        estimators = [(name, model) for name, model in base_models.items()]
        
        # Meta-learner with regularization
        meta_learner = LogisticRegression(
            C=0.1,  # Strong regularization
            penalty='elasticnet',
            l1_ratio=0.5,
            solver='saga',
            max_iter=2000,
            random_state=self.random_state
        )
        
        # Create stacking classifier
        stacking_clf = StackingClassifier(
            estimators=estimators,
            final_estimator=meta_learner,
            cv=StratifiedKFold(n_splits=8, shuffle=True, random_state=self.random_state),
            stack_method='predict_proba',
            n_jobs=-1,
            verbose=1
        )
        
        print("   🔧 Training meta-ensemble...")
        stacking_clf.fit(X_train, y_train)
        
        return {'MetaEnsemble': stacking_clf}
        
    def create_weighted_ensemble(self, models, X_val, y_val):
        """Create weighted ensemble based on validation performance"""
        print("⚖️ Creating weighted ensemble...")
        
        # Calculate weights based on validation accuracy
        weights = {}
        total_weight = 0
        
        for name, model in models.items():
            try:
                val_pred = model.predict(X_val)
                accuracy = accuracy_score(y_val, val_pred)
                # Square the accuracy to emphasize better models
                weight = accuracy ** 2
                weights[name] = weight
                total_weight += weight
                print(f"   📊 {name}: {accuracy:.4f} accuracy, weight: {weight:.4f}")
            except Exception as e:
                print(f"   ❌ {name} failed validation: {str(e)}")
                weights[name] = 0
        
        # Normalize weights
        for name in weights:
            weights[name] = weights[name] / total_weight if total_weight > 0 else 0
            
        self.ensemble_weights = weights
        return weights
        
    def predict_weighted_ensemble(self, models, X):
        """Make predictions using weighted ensemble"""
        if not hasattr(self, 'ensemble_weights'):
            # Default equal weights
            self.ensemble_weights = {name: 1.0/len(models) for name in models.keys()}
        
        # Collect predictions from all models
        predictions = []
        model_names = []
        
        for name, model in models.items():
            try:
                pred_proba = model.predict_proba(X)
                predictions.append(pred_proba * self.ensemble_weights[name])
                model_names.append(name)
            except Exception as e:
                print(f"⚠️ {name} prediction failed: {str(e)}")
                continue
        
        if not predictions:
            raise ValueError("No models could make predictions!")
        
        # Weighted average
        ensemble_proba = np.sum(predictions, axis=0)
        ensemble_pred = np.argmax(ensemble_proba, axis=1)
        
        return ensemble_pred, ensemble_proba
        
    def comprehensive_evaluation(self, models, X_test, y_test):
        """Comprehensive evaluation of all models"""
        print("📊 Comprehensive evaluation...")
        
        results = {}
        
        # Individual model results
        for name, model in models.items():
            try:
                pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, pred)
                results[name] = {
                    'accuracy': accuracy,
                    'predictions': pred,
                    'type': 'individual'
                }
                print(f"   ✅ {name}: {accuracy:.4f}")
            except Exception as e:
                print(f"   ❌ {name}: {str(e)}")
                continue
        
        # Weighted ensemble
        try:
            weighted_pred, weighted_proba = self.predict_weighted_ensemble(models, X_test)
            weighted_accuracy = accuracy_score(y_test, weighted_pred)
            results['WeightedEnsemble'] = {
                'accuracy': weighted_accuracy,
                'predictions': weighted_pred,
                'probabilities': weighted_proba,
                'type': 'ensemble'
            }
            print(f"   🏆 WeightedEnsemble: {weighted_accuracy:.4f}")
        except Exception as e:
            print(f"   ⚠️ WeightedEnsemble failed: {str(e)}")
            
        return results
        
    def generate_ultimate_report(self, results, y_test):
        """Generate the ultimate performance report"""
        print("📋 Generating ultimate performance report...")
        
        # Sort results by accuracy
        sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        
        # Best model analysis
        best_name, best_result = sorted_results[0]
        best_pred = best_result['predictions']
        
        # Classification report
        class_report = classification_report(
            y_test, best_pred,
            target_names=self.class_names,
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_test, best_pred)
        
        # Create comprehensive results DataFrame
        results_summary = []
        
        for name, result in sorted_results:
            results_summary.append({
                'Model': name,
                'Accuracy': result['accuracy'],
                'Type': result['type'],
                'Improvement_vs_Baseline': (result['accuracy'] - 0.701) * 100
            })
        
        results_df = pd.DataFrame(results_summary)
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_path = f"results/ultimate_ensemble/ultimate_results_{timestamp}.csv"
        results_df.to_csv(results_path, index=False)
        
        # Create visualizations
        self._create_ultimate_visualizations(results_df, cm, class_report, timestamp)
        
        # Print ultimate summary
        print("\n🏆 ULTIMATE EXOPLANET CLASSIFIER RESULTS")
        print("=" * 70)
        print(f"{'Model':<25} {'Accuracy':<12} {'Type':<12} {'Improvement':<12}")
        print("-" * 70)
        
        for _, row in results_df.iterrows():
            print(f"{row['Model']:<25} {row['Accuracy']:<12.4f} {row['Type']:<12} "
                  f"+{row['Improvement_vs_Baseline']:<11.2f}%")
        
        print("-" * 70)
        
        ultimate_accuracy = results_df.iloc[0]['Accuracy']
        ultimate_model = results_df.iloc[0]['Model']
        
        print(f"\n🥇 ULTIMATE CHAMPION: {ultimate_model}")
        print(f"🎯 ULTIMATE ACCURACY: {ultimate_accuracy:.4f}")
        print(f"🚀 TOTAL IMPROVEMENT: +{(ultimate_accuracy - 0.701)*100:.2f}% over baseline")
        
        # Performance by class
        print(f"\n📈 Per-class Performance (Best Model):")
        for class_name in self.class_names:
            if class_name in class_report:
                metrics = class_report[class_name]
                print(f"   {class_name:<15} Precision: {metrics['precision']:.3f}  "
                      f"Recall: {metrics['recall']:.3f}  F1: {metrics['f1-score']:.3f}")
        
        print(f"\n💾 Results saved to: {results_path}")
        
        return results_df
        
    def _create_ultimate_visualizations(self, results_df, cm, class_report, timestamp):
        """Create ultimate performance visualizations"""
        
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Model accuracy comparison
        plt.subplot(2, 3, 1)
        models = results_df['Model']
        accuracies = results_df['Accuracy']
        colors = ['gold' if i == 0 else 'skyblue' for i in range(len(models))]
        
        bars = plt.barh(range(len(models)), accuracies, color=colors)
        plt.yticks(range(len(models)), models)
        plt.xlabel('Accuracy')
        plt.title('Ultimate Model Comparison')
        plt.xlim(min(accuracies) * 0.95, max(accuracies) * 1.02)
        
        # Add accuracy labels
        for i, (bar, acc) in enumerate(zip(bars, accuracies)):
            plt.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                    f'{acc:.4f}', va='center', ha='left', fontweight='bold' if i == 0 else 'normal')
        
        # 2. Improvement over baseline
        plt.subplot(2, 3, 2)
        improvements = results_df['Improvement_vs_Baseline']
        bars = plt.bar(range(len(models)), improvements, color=['gold' if i == 0 else 'lightgreen' for i in range(len(models))])
        plt.xticks(range(len(models)), models, rotation=45, ha='right')
        plt.ylabel('Improvement (%)')
        plt.title('Improvement over Baseline (70.1%)')
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        
        # 3. Confusion Matrix
        plt.subplot(2, 3, 3)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Confusion Matrix (Best Model)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # 4. Per-class Performance
        plt.subplot(2, 3, 4)
        class_metrics = ['precision', 'recall', 'f1-score']
        class_data = []
        
        for class_name in self.class_names:
            if class_name in class_report:
                class_data.append([
                    class_report[class_name]['precision'],
                    class_report[class_name]['recall'],
                    class_report[class_name]['f1-score']
                ])
        
        class_data = np.array(class_data)
        x = np.arange(len(self.class_names))
        width = 0.25
        
        for i, metric in enumerate(class_metrics):
            plt.bar(x + i*width, class_data[:, i], width, label=metric.title(), alpha=0.8)
        
        plt.xlabel('Classes')
        plt.ylabel('Score')
        plt.title('Per-class Performance Metrics')
        plt.xticks(x + width, self.class_names)
        plt.legend()
        plt.ylim(0, 1)
        
        # 5. Model Type Distribution
        plt.subplot(2, 3, 5)
        type_counts = results_df['Type'].value_counts()
        plt.pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%', startangle=90)
        plt.title('Model Type Distribution')
        
        # 6. Accuracy Distribution
        plt.subplot(2, 3, 6)
        plt.hist(accuracies, bins=max(3, len(accuracies)//2), alpha=0.7, color='lightblue', edgecolor='black')
        plt.xlabel('Accuracy')
        plt.ylabel('Number of Models')
        plt.title('Accuracy Distribution')
        plt.axvline(x=accuracies.iloc[0], color='gold', linestyle='--', linewidth=2, label='Best Model')
        plt.axvline(x=0.701, color='red', linestyle='--', linewidth=2, label='Baseline')
        plt.legend()
        
        plt.suptitle('🏆 Ultimate Exoplanet Classifier Performance Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'results/ultimate_ensemble/ultimate_analysis_{timestamp}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("📊 Ultimate visualizations saved!")
        
    def run_ultimate_ensemble(self):
        """Run the complete ultimate ensemble pipeline"""
        print("\n🏆 Starting Ultimate Ensemble Creation")
        print("=" * 50)
        
        # 1. Load best models
        loaded_models, preprocessing = self.load_best_models()
        if not loaded_models:
            print("❌ No models could be loaded. Run training first!")
            return None
        
        # 2. Load test data
        print("📥 Loading test data...")
        features = pd.read_csv("data/processed/features.csv")
        labels = pd.read_csv("data/processed/labels.csv")
        
        # Create enhanced features
        enhanced_features = self.create_hybrid_features(features)
        
        # Split data the same way
        from sklearn.model_selection import train_test_split
        X_temp, X_test, y_temp, y_test = train_test_split(
            enhanced_features, labels['label'], test_size=0.2, 
            stratify=labels['label'], random_state=self.random_state
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=self.random_state
        )
        
        # Apply preprocessing
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        X_train_selected = self.feature_selector.transform(X_train_scaled)
        X_val_selected = self.feature_selector.transform(X_val_scaled)
        X_test_selected = self.feature_selector.transform(X_test_scaled)
        
        # Encode labels
        y_train_encoded = self.label_encoder.transform(y_train)
        y_val_encoded = self.label_encoder.transform(y_val)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        print(f"📊 Final data - Train: {len(X_train_selected)}, Val: {len(X_val_selected)}, Test: {len(X_test_selected)}")
        
        # 3. Train additional specialized models
        additional_models = self.train_additional_models(X_train_selected, y_train_encoded)
        
        # 4. Combine all models
        all_models = {**loaded_models, **additional_models}
        
        # 5. Create meta-ensemble
        meta_models = self.create_meta_ensemble(all_models, X_train_selected, y_train_encoded)
        all_models.update(meta_models)
        
        # 6. Calculate ensemble weights
        self.create_weighted_ensemble(all_models, X_val_selected, y_val_encoded)
        
        # 7. Final evaluation
        results = self.comprehensive_evaluation(all_models, X_test_selected, y_test_encoded)
        
        # 8. Generate ultimate report
        results_df = self.generate_ultimate_report(results, y_test_encoded)
        
        print("\n✅ Ultimate Ensemble Complete!")
        return results_df

def main():
    """Run ultimate ensemble"""
    ensemble = UltimateExoplanetEnsemble()
    results = ensemble.run_ultimate_ensemble()
    return results

if __name__ == "__main__":
    results = main()