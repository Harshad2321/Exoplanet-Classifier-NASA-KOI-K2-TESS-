#!/usr/bin/env python3
"""
🤖 NASA Smart Exoplanet Classifier with Automatic Model Selection
Intelligent system that automatically selects the best AI model based on data characteristics.

Author: NASA Space Apps Challenge Team
Date: October 1, 2025
Challenge: A World Away - Hunting for Exoplanets with AI
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.impute import SimpleImputer
import joblib
import json
import warnings
from pathlib import Path
from datetime import datetime
import logging
from collections import Counter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('smart_exoplanet_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class SmartNASAExoplanetClassifier:
    """
    🚀 Smart NASA Exoplanet Classification System with Automatic Model Selection
    
    This intelligent system analyzes data characteristics and automatically selects
    the optimal AI model for exoplanet classification.
    """
    
    def __init__(self, random_state=2025):
        """Initialize the smart classifier"""
        self.random_state = random_state
        self.models = {}
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        self.training_history = {}
        self.data_characteristics = {}
        self.selected_model = None
        self.selection_reason = ""
        
        # Set random seeds for reproducibility
        np.random.seed(self.random_state)
        
        logger.info("🤖 Smart NASA Exoplanet Classifier initialized")
        logger.info("   ✨ Automatic model selection enabled!")
    
    def analyze_data_characteristics(self, df):
        """
        🔍 Analyze data characteristics to determine optimal model
        
        Args:
            df: Input dataframe
            
        Returns:
            dict: Data characteristics analysis
        """
        logger.info("🔍 Analyzing data characteristics for optimal model selection...")
        
        # Basic dataset info
        n_samples, n_features = df.shape
        target_col = 'koi_disposition' if 'koi_disposition' in df.columns else df.columns[-1]
        feature_cols = [col for col in df.columns if col != target_col and col not in ['kepid', 'kepler_name']]
        
        # Calculate characteristics
        characteristics = {
            'n_samples': n_samples,
            'n_features': len(feature_cols),
            'missing_ratio': df[feature_cols].isnull().sum().sum() / (n_samples * len(feature_cols)),
            'class_distribution': dict(df[target_col].value_counts()),
            'imbalance_ratio': df[target_col].value_counts().min() / df[target_col].value_counts().max(),
            'numeric_features': len(df[feature_cols].select_dtypes(include=[np.number]).columns),
            'categorical_features': len(df[feature_cols].select_dtypes(exclude=[np.number]).columns),
            'outlier_ratio': 0,  # Will calculate below
            'noise_level': 0,    # Will calculate below
            'feature_correlation': 0  # Will calculate below
        }
        
        # Calculate outlier ratio using IQR method
        numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns
        outlier_count = 0
        total_numeric_values = 0
        
        for col in numeric_cols:
            if not df[col].empty:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
                outlier_count += outliers
                total_numeric_values += len(df[col].dropna())
        
        if total_numeric_values > 0:
            characteristics['outlier_ratio'] = outlier_count / total_numeric_values
        
        # Estimate noise level (coefficient of variation for numeric features)
        cv_values = []
        for col in numeric_cols:
            if not df[col].empty and df[col].std() > 0:
                cv = df[col].std() / abs(df[col].mean()) if df[col].mean() != 0 else 0
                cv_values.append(cv)
        
        characteristics['noise_level'] = np.mean(cv_values) if cv_values else 0
        
        # Calculate average feature correlation
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr().abs()
            # Get upper triangle of correlation matrix (excluding diagonal)
            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            characteristics['feature_correlation'] = upper_triangle.stack().mean()
        
        self.data_characteristics = characteristics
        
        # Log characteristics
        logger.info(f"📊 Dataset Analysis Results:")
        logger.info(f"   📏 Samples: {n_samples:,}, Features: {len(feature_cols)}")
        logger.info(f"   🕳️ Missing data: {characteristics['missing_ratio']:.1%}")
        logger.info(f"   ⚖️ Class imbalance ratio: {characteristics['imbalance_ratio']:.3f}")
        logger.info(f"   🎯 Outlier ratio: {characteristics['outlier_ratio']:.1%}")
        logger.info(f"   📡 Noise level: {characteristics['noise_level']:.3f}")
        logger.info(f"   🔗 Feature correlation: {characteristics['feature_correlation']:.3f}")
        
        return characteristics
    
    def select_optimal_model(self):
        """
        🎯 Automatically select the optimal model based on data characteristics
        
        Returns:
            str: Selected model name and reason
        """
        logger.info("🎯 Selecting optimal AI model based on data characteristics...")
        
        char = self.data_characteristics
        
        # Decision tree for model selection
        if char['n_samples'] < 1000:
            # Small dataset
            if char['noise_level'] > 0.5:
                selected = 'extra_trees'
                reason = "Small noisy dataset - Extra Trees handles noise well with limited data"
            elif char['imbalance_ratio'] < 0.3:
                selected = 'balanced_random_forest'
                reason = "Small imbalanced dataset - Balanced Random Forest handles class imbalance"
            else:
                selected = 'random_forest'
                reason = "Small clean dataset - Random Forest provides stable predictions"
                
        elif char['n_samples'] < 5000:
            # Medium dataset
            if char['outlier_ratio'] > 0.1 or char['noise_level'] > 0.3:
                selected = 'extra_trees'
                reason = "Medium noisy dataset - Extra Trees robust to outliers and noise"
            elif char['feature_correlation'] > 0.7:
                selected = 'gradient_boosting'
                reason = "High feature correlation - Gradient Boosting handles redundant features"
            else:
                selected = 'ensemble'
                reason = "Medium balanced dataset - Ensemble combines multiple model strengths"
                
        else:
            # Large dataset
            if char['missing_ratio'] > 0.2:
                selected = 'random_forest'
                reason = "Large dataset with missing values - Random Forest handles missing data well"
            elif char['imbalance_ratio'] < 0.2:
                selected = 'balanced_ensemble'
                reason = "Large imbalanced dataset - Balanced ensemble addresses class imbalance"
            elif char['noise_level'] > 0.4:
                selected = 'extra_trees'
                reason = "Large noisy dataset - Extra Trees excels with noise in large datasets"
            else:
                selected = 'ensemble'
                reason = "Large clean dataset - Ensemble maximizes accuracy with sufficient data"
        
        # Special cases
        if char['n_features'] > 50:
            if selected not in ['extra_trees', 'random_forest']:
                selected = 'random_forest'
                reason = "High-dimensional data - Random Forest handles many features efficiently"
        
        if char['categorical_features'] > char['numeric_features']:
            if selected not in ['random_forest', 'extra_trees']:
                selected = 'random_forest'
                reason = "Mostly categorical features - Tree-based models handle categories well"
        
        self.selected_model = selected
        self.selection_reason = reason
        
        logger.info(f"🎯 Selected Model: {selected.replace('_', ' ').title()}")
        logger.info(f"📝 Reason: {reason}")
        
        return selected, reason
    
    def build_all_models(self):
        """Build all available models for comparison and ensemble"""
        logger.info("🤖 Building comprehensive AI model suite...")
        
        # Random Forest - Stable and interpretable
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # Balanced Random Forest - For imbalanced data
        balanced_rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=self.random_state,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        # Extra Trees - Noise resistant
        et_model = ExtraTreesClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # Gradient Boosting - Sequential learning
        gb_model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=10,
            random_state=self.random_state
        )
        
        # Standard Ensemble
        ensemble_model = VotingClassifier(
            estimators=[
                ('rf', rf_model),
                ('et', et_model)
            ],
            voting='soft'
        )
        
        # Balanced Ensemble
        balanced_ensemble_model = VotingClassifier(
            estimators=[
                ('balanced_rf', balanced_rf_model),
                ('et', et_model),
                ('gb', gb_model)
            ],
            voting='soft'
        )
        
        self.models = {
            'random_forest': rf_model,
            'balanced_random_forest': balanced_rf_model,
            'extra_trees': et_model,
            'gradient_boosting': gb_model,
            'ensemble': ensemble_model,
            'balanced_ensemble': balanced_ensemble_model
        }
        
        logger.info(f"✅ Built {len(self.models)} AI models")
        return self.models
    
    def get_optimal_model(self):
        """Get the automatically selected optimal model"""
        if self.selected_model and self.selected_model in self.models:
            return self.models[self.selected_model]
        else:
            logger.warning("⚠️ No model selected, using ensemble as default")
            return self.models.get('ensemble', list(self.models.values())[0])
    
    def preprocess_data(self, df):
        """
        Preprocess the data for training
        
        Args:
            df: Raw dataframe
            
        Returns:
            X: Processed features
            y: Encoded target labels
        """
        logger.info("🔄 Starting intelligent data preprocessing...")
        
        # Identify feature and target columns
        target_col = 'koi_disposition' if 'koi_disposition' in df.columns else df.columns[-1]
        feature_cols = [col for col in df.columns if col != target_col and col not in ['kepid', 'kepler_name']]
        
        # Extract features and target
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # Store feature names
        self.feature_names = feature_cols
        
        # Handle missing values intelligently
        logger.info("🔧 Intelligent missing value handling...")
        missing_before = X.isnull().sum().sum()
        
        # Use median for high-noise data, mean for clean data
        strategy = 'median' if self.data_characteristics.get('noise_level', 0) > 0.3 else 'mean'
        self.imputer = SimpleImputer(strategy=strategy)
        X_imputed = self.imputer.fit_transform(X)
        X = pd.DataFrame(X_imputed, columns=feature_cols, index=X.index)
        logger.info(f"   Filled {missing_before} missing values using {strategy} strategy")
        
        # Feature engineering
        logger.info("⚙️ Engineering astronomical features...")
        X = self._engineer_features(X)
        
        # Smart scaling based on data characteristics
        logger.info("📏 Applying intelligent feature scaling...")
        X_scaled = self.scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        # Encode target labels
        logger.info("🏷️ Encoding target labels...")
        y_encoded = self.label_encoder.fit_transform(y)
        
        logger.info(f"✅ Preprocessing complete:")
        logger.info(f"   Features shape: {X.shape}")
        logger.info(f"   Target classes: {list(self.label_encoder.classes_)}")
        
        return X, y_encoded
    
    def _engineer_features(self, X):
        """Engineer additional features based on astronomical knowledge"""
        
        # Planetary characteristics
        if 'koi_period' in X.columns and 'koi_prad' in X.columns:
            X['planet_mass_proxy'] = X['koi_prad'] ** 2.06  # Mass-radius relation
        
        if 'koi_teq' in X.columns and 'koi_steff' in X.columns:
            X['temp_ratio'] = X['koi_teq'] / X['koi_steff']
        
        # Orbital characteristics  
        if 'koi_period' in X.columns and 'koi_smass' in X.columns:
            X['orbital_velocity'] = (2 * np.pi * X['koi_dor'] * X['koi_srad']) / X['koi_period']
        
        # Habitability indicators
        if 'koi_teq' in X.columns:
            X['habitable_zone'] = ((X['koi_teq'] >= 200) & (X['koi_teq'] <= 400)).astype(int)
        
        # Detection difficulty
        if 'koi_prad' in X.columns and 'koi_srad' in X.columns:
            X['transit_depth'] = (X['koi_prad'] / (109 * X['koi_srad'])) ** 2  # Relative to Jupiter
        
        return X
    
    def smart_train(self, df, test_size=0.2):
        """
        🚀 Intelligent training with automatic model selection
        
        Args:
            df: Input dataframe
            test_size: Fraction of data for testing
            
        Returns:
            dict: Training results with model selection info
        """
        logger.info("🚀 Starting Smart NASA AI Training Pipeline...")
        
        # Step 1: Analyze data characteristics
        self.analyze_data_characteristics(df)
        
        # Step 2: Select optimal model
        selected_model, reason = self.select_optimal_model()
        
        # Step 3: Build all models
        self.build_all_models()
        
        # Step 4: Preprocess data
        X, y = self.preprocess_data(df)
        
        # Step 5: Train the selected model + comparison models
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        logger.info(f"📊 Training set: {len(X_train)} samples")
        logger.info(f"📊 Test set: {len(X_test)} samples")
        
        results = {}
        
        # Train the selected model first
        logger.info(f"🎯 Training SELECTED model: {selected_model.replace('_', ' ').title()}")
        model = self.models[selected_model]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        cv_scores = cross_val_score(
            model, X_train, y_train, 
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
            scoring='accuracy',
            n_jobs=-1
        )
        
        results[selected_model] = {
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'is_selected': True,
            'selection_reason': reason
        }
        
        logger.info(f"   🎯 SELECTED {selected_model}: {accuracy:.1%} accuracy (CV: {cv_scores.mean():.1%} ± {cv_scores.std():.1%})")
        
        # Train comparison models for validation
        comparison_models = ['random_forest', 'extra_trees', 'ensemble']
        for model_name in comparison_models:
            if model_name != selected_model and model_name in self.models:
                logger.info(f"📊 Training comparison: {model_name.replace('_', ' ').title()}")
                
                model = self.models[model_name]
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                cv_scores = cross_val_score(
                    model, X_train, y_train, 
                    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
                    scoring='accuracy',
                    n_jobs=-1
                )
                
                results[model_name] = {
                    'accuracy': accuracy,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'is_selected': False
                }
                
                logger.info(f"   📊 {model_name}: {accuracy:.1%} accuracy (CV: {cv_scores.mean():.1%} ± {cv_scores.std():.1%})")
        
        # Store results
        self.training_history = results
        
        # Validate selection
        selected_accuracy = results[selected_model]['accuracy']
        best_comparison = max([r['accuracy'] for k, r in results.items() if not r.get('is_selected', False)], default=0)
        
        if selected_accuracy >= best_comparison:
            logger.info(f"✅ Smart selection VALIDATED: {selected_model} performs best!")
        else:
            logger.info(f"⚠️  Smart selection suboptimal, but choice was data-driven: {reason}")
        
        # Summary report
        logger.info("📋 Smart Training Summary:")
        logger.info(f"   🎯 Selected Model: {selected_model.replace('_', ' ').title()}")
        logger.info(f"   📝 Selection Reason: {reason}")
        logger.info(f"   🏆 Selected Accuracy: {selected_accuracy:.1%}")
        logger.info(f"   📊 Dataset: {len(df)} samples, {len(self.feature_names)} features")
        
        return results
    
    def predict_smart(self, X):
        """Make predictions using the automatically selected model"""
        optimal_model = self.get_optimal_model()
        return optimal_model.predict(X)
    
    def predict_proba_smart(self, X):
        """Get prediction probabilities using the automatically selected model"""
        optimal_model = self.get_optimal_model()
        return optimal_model.predict_proba(X)
    
    def save_smart_model(self, filepath='nasa_smart_classifier.joblib'):
        """Save the smart classifier with selection logic"""
        model_data = {
            'models': self.models,
            'selected_model': self.selected_model,
            'selection_reason': self.selection_reason,
            'data_characteristics': self.data_characteristics,
            'scaler': self.scaler,
            'imputer': self.imputer,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'training_history': self.training_history
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"💾 Smart classifier saved to {filepath}")
        logger.info(f"   🎯 Selected model: {self.selected_model}")
    
    def load_smart_model(self, filepath='nasa_smart_classifier.joblib'):
        """Load the smart classifier"""
        model_data = joblib.load(filepath)
        
        self.models = model_data['models']
        self.selected_model = model_data['selected_model']
        self.selection_reason = model_data['selection_reason']
        self.data_characteristics = model_data['data_characteristics']
        self.scaler = model_data['scaler']
        self.imputer = model_data['imputer']
        self.label_encoder = model_data['label_encoder']
        self.feature_names = model_data['feature_names']
        self.training_history = model_data['training_history']
        
        logger.info(f"📥 Smart classifier loaded from {filepath}")
        logger.info(f"   🎯 Selected model: {self.selected_model}")
        logger.info(f"   📝 Selection reason: {self.selection_reason}")
    
    def generate_smart_report(self):
        """Generate comprehensive smart training report"""
        logger.info("📋 Generating Smart AI Report...")
        
        report = {
            'training_date': datetime.now().isoformat(),
            'challenge': 'NASA Space Apps Challenge 2025 - Smart AI Model Selection',
            'data_characteristics': self.data_characteristics,
            'model_selection': {
                'selected_model': self.selected_model,
                'selection_reason': self.selection_reason,
                'available_models': list(self.models.keys())
            },
            'model_performance': self.training_history,
            'feature_info': {
                'total_features': len(self.feature_names),
                'feature_names': self.feature_names,
                'target_classes': self.label_encoder.classes_.tolist()
            }
        }
        
        # Save report
        report_file = f'smart_ai_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"📋 Smart AI report saved to {report_file}")
        return report

# Example usage and demo
def demo_smart_classifier():
    """Demonstrate the smart classifier with different data scenarios"""
    logger.info("🎬 Demonstrating Smart NASA Exoplanet Classifier...")
    
    # Initialize smart classifier
    smart_classifier = SmartNASAExoplanetClassifier()
    
    # Create different data scenarios for demonstration
    scenarios = [
        {"name": "Small Clean Dataset", "n_samples": 800, "noise": 0.1, "missing": 0.05},
        {"name": "Medium Noisy Dataset", "n_samples": 3000, "noise": 0.4, "missing": 0.15},
        {"name": "Large Imbalanced Dataset", "n_samples": 8000, "noise": 0.2, "missing": 0.1}
    ]
    
    for scenario in scenarios:
        logger.info(f"\n🎯 Testing Scenario: {scenario['name']}")
        
        # Generate synthetic data matching scenario
        df = create_test_dataset(**scenario)
        
        # Train with smart selection
        results = smart_classifier.smart_train(df)
        
        logger.info(f"✅ Scenario '{scenario['name']}' completed")
        print("-" * 60)

def create_test_dataset(name, n_samples, noise, missing):
    """Create test dataset with specified characteristics"""
    np.random.seed(42)
    
    # Create base features
    data = {
        'koi_period': np.random.lognormal(2, 1, n_samples),
        'koi_prad': np.random.lognormal(0, 0.5, n_samples),
        'koi_teq': 200 + np.random.exponential(200, n_samples),
        'koi_steff': 5000 + np.random.normal(0, 1000, n_samples),
        'koi_smass': np.random.lognormal(0, 0.3, n_samples),
        'koi_srad': np.random.lognormal(0, 0.2, n_samples),
        'koi_dor': np.random.uniform(2, 50, n_samples)
    }
    
    # Add noise
    for key in data:
        noise_factor = np.random.normal(1, noise, n_samples)
        data[key] = data[key] * noise_factor
    
    # Create target based on realistic rules
    target = []
    for i in range(n_samples):
        if data['koi_period'][i] < 100 and data['koi_prad'][i] < 4:
            if np.random.random() < 0.7:
                target.append('CONFIRMED')
            else:
                target.append('CANDIDATE')
        else:
            if np.random.random() < 0.8:
                target.append('FALSE POSITIVE')
            else:
                target.append('CANDIDATE')
    
    # Create DataFrame
    df = pd.DataFrame(data)
    df['koi_disposition'] = target
    
    # Add missing values
    for col in df.columns[:-1]:  # Don't add missing to target
        missing_indices = np.random.choice(n_samples, int(n_samples * missing), replace=False)
        df.loc[missing_indices, col] = np.nan
    
    return df

if __name__ == "__main__":
    demo_smart_classifier()