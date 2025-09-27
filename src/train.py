"""
Model Training Pipeline for NASA Space Apps Challenge 2025
"A World Away: Hunting for Exoplanets with AI"

This module handles machine learning model training, evaluation, and selection
for the exoplanet classification task.
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import warnings
import logging
from datetime import datetime

# Scikit-learn imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

# XGBoost import (with fallback)
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️  XGBoost not available. Using RandomForest as advanced model.")

# Import our preprocessor
from preprocess import ExoplanetDataPreprocessor

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExoplanetModelTrainer:
    """
    Comprehensive machine learning pipeline for exoplanet classification.
    
    Handles model training, evaluation, hyperparameter tuning, and model selection.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the model trainer.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.model_results = {}
        self.best_model = None
        self.label_encoder = LabelEncoder()
        self.preprocessor = None
        
        # Initialize models
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize the machine learning models to train."""
        
        # Baseline model - Logistic Regression
        self.models['logistic_regression'] = {
            'model': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                class_weight='balanced'
            ),
            'params': {
                'C': [0.1, 1.0, 10.0],
                'solver': ['lbfgs', 'liblinear'],
                'penalty': ['l2']
            },
            'description': 'Logistic Regression (Baseline)'
        }
        
        # Tree-based model - Random Forest
        self.models['random_forest'] = {
            'model': RandomForestClassifier(
                random_state=self.random_state,
                class_weight='balanced',
                n_jobs=-1
            ),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [5, 10],
                'min_samples_leaf': [2, 4]
            },
            'description': 'Random Forest Classifier'
        }
        
        # Support Vector Machine
        self.models['svm'] = {
            'model': SVC(
                random_state=self.random_state,
                class_weight='balanced',
                probability=True
            ),
            'params': {
                'C': [0.1, 1.0, 10.0],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto']
            },
            'description': 'Support Vector Machine'
        }
        
        # Advanced model - XGBoost (if available)
        if XGBOOST_AVAILABLE:
            self.models['xgboost'] = {
                'model': XGBClassifier(
                    random_state=self.random_state,
                    eval_metric='mlogloss',
                    verbosity=0
                ),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0]
                },
                'description': 'XGBoost Classifier'
            }
    
    def load_preprocessed_data(self, data_dir: str = "data/processed") -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Load preprocessed data from files.
        
        Args:
            data_dir: Directory containing preprocessed data files
            
        Returns:
            Tuple of (features, labels, sources)
        """
        data_path = Path(data_dir)
        
        try:
            X = pd.read_csv(data_path / "features.csv")
            y = pd.read_csv(data_path / "labels.csv")['label']
            sources = pd.read_csv(data_path / "sources.csv")['source']
            
            logger.info(f"✅ Loaded preprocessed data: {X.shape[0]:,} samples, {X.shape[1]} features")
            return X, y, sources
            
        except FileNotFoundError:
            logger.warning("❌ Preprocessed data not found. Running preprocessing...")
            return self._run_preprocessing()
    
    def _run_preprocessing(self) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Run preprocessing pipeline if data not available.
        
        Returns:
            Tuple of (features, labels, sources)
        """
        self.preprocessor = ExoplanetDataPreprocessor()
        datasets = self.preprocessor.load_datasets()
        X, y, sources = self.preprocessor.combine_datasets(datasets)
        return X, y, sources
    
    def split_data(self, X: pd.DataFrame, y: pd.Series, 
                   test_size: float = 0.2, val_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Split data into train/validation/test sets.
        
        Args:
            X: Features
            y: Labels
            test_size: Proportion of data for test set
            val_size: Proportion of remaining data for validation set
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=self.random_state
        )
        
        # Second split: separate train and validation from remaining data
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, stratify=y_temp, 
            random_state=self.random_state
        )
        
        logger.info(f"📊 Data split:")
        logger.info(f"   Training: {X_train.shape[0]:,} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
        logger.info(f"   Validation: {X_val.shape[0]:,} samples ({X_val.shape[0]/len(X)*100:.1f}%)")
        logger.info(f"   Test: {X_test.shape[0]:,} samples ({X_test.shape[0]/len(X)*100:.1f}%)")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_model(self, model_name: str, X_train: pd.DataFrame, y_train: pd.Series,
                   X_val: pd.DataFrame = None, y_val: pd.Series = None,
                   use_grid_search: bool = True) -> Dict[str, Any]:
        """
        Train a single model with hyperparameter tuning.
        
        Args:
            model_name: Name of the model to train
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            use_grid_search: Whether to perform hyperparameter tuning
            
        Returns:
            Dictionary with training results
        """
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")
        
        model_config = self.models[model_name]
        base_model = model_config['model']
        param_grid = model_config['params']
        description = model_config['description']
        
        logger.info(f"🚀 Training {description}...")
        
        start_time = datetime.now()
        
        if use_grid_search and len(param_grid) > 0:
            # Hyperparameter tuning with cross-validation
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
            
            grid_search = GridSearchCV(
                base_model,
                param_grid,
                cv=cv,
                scoring='f1_macro',
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            cv_score = grid_search.best_score_
            
            logger.info(f"   ✅ Best parameters: {best_params}")
            logger.info(f"   ✅ CV F1-score: {cv_score:.4f}")
            
        else:
            # Train with default parameters
            best_model = base_model
            best_model.fit(X_train, y_train)
            best_params = {}
            cv_score = None
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Make predictions
        y_train_pred = best_model.predict(X_train)
        y_train_proba = best_model.predict_proba(X_train)
        
        train_metrics = self._calculate_metrics(y_train, y_train_pred, y_train_proba)
        
        val_metrics = {}
        if X_val is not None and y_val is not None:
            y_val_pred = best_model.predict(X_val)
            y_val_proba = best_model.predict_proba(X_val)
            val_metrics = self._calculate_metrics(y_val, y_val_pred, y_val_proba)
        
        # Store results
        results = {
            'model': best_model,
            'model_name': model_name,
            'description': description,
            'best_params': best_params,
            'cv_score': cv_score,
            'training_time': training_time,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'classes': best_model.classes_
        }
        
        self.model_results[model_name] = results
        
        logger.info(f"   ⏱️  Training time: {training_time:.2f} seconds")
        logger.info(f"   📊 Train F1-score: {train_metrics['f1_macro']:.4f}")
        if val_metrics:
            logger.info(f"   📊 Val F1-score: {val_metrics['f1_macro']:.4f}")
        
        return results
    
    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray, 
                          y_proba: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        # Add AUC-ROC for multiclass
        try:
            if len(np.unique(y_true)) > 2:
                metrics['auc_roc_macro'] = roc_auc_score(y_true, y_proba, average='macro', multi_class='ovr')
                metrics['auc_roc_weighted'] = roc_auc_score(y_true, y_proba, average='weighted', multi_class='ovr')
            else:
                metrics['auc_roc'] = roc_auc_score(y_true, y_proba[:, 1])
        except:
            metrics['auc_roc_macro'] = 0.0
            metrics['auc_roc_weighted'] = 0.0
        
        return metrics
    
    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                        X_val: pd.DataFrame = None, y_val: pd.Series = None) -> Dict[str, Dict[str, Any]]:
        """
        Train all available models.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Dictionary of all model results
        """
        logger.info("🚀 Training all models...")
        logger.info("=" * 60)
        
        for model_name in self.models.keys():
            try:
                self.train_model(model_name, X_train, y_train, X_val, y_val)
                print()  # Empty line for readability
            except Exception as e:
                logger.error(f"❌ Failed to train {model_name}: {e}")
        
        return self.model_results
    
    def select_best_model(self, metric: str = 'f1_macro') -> Dict[str, Any]:
        """
        Select the best model based on validation performance.
        
        Args:
            metric: Metric to use for model selection
            
        Returns:
            Best model results
        """
        if not self.model_results:
            raise ValueError("No models trained yet. Call train_all_models() first.")
        
        best_score = -1
        best_model_name = None
        
        for model_name, results in self.model_results.items():
            # Use validation metrics if available, otherwise training metrics
            metrics = results.get('val_metrics', results.get('train_metrics', {}))
            
            if metric in metrics:
                score = metrics[metric]
                if score > best_score:
                    best_score = score
                    best_model_name = model_name
        
        if best_model_name:
            self.best_model = self.model_results[best_model_name]
            logger.info(f"🏆 Best model: {self.best_model['description']}")
            logger.info(f"   {metric}: {best_score:.4f}")
            return self.best_model
        else:
            raise ValueError(f"Could not select best model using metric: {metric}")
    
    def evaluate_on_test(self, X_test: pd.DataFrame, y_test: pd.Series,
                        model_results: Dict[str, Any] = None) -> Dict[str, float]:
        """
        Evaluate model on test set.
        
        Args:
            X_test: Test features
            y_test: Test labels
            model_results: Model results to evaluate (uses best model if None)
            
        Returns:
            Test set metrics
        """
        if model_results is None:
            if self.best_model is None:
                raise ValueError("No best model selected. Call select_best_model() first.")
            model_results = self.best_model
        
        model = model_results['model']
        model_name = model_results['description']
        
        logger.info(f"🧪 Testing {model_name} on test set...")
        
        # Make predictions
        y_test_pred = model.predict(X_test)
        y_test_proba = model.predict_proba(X_test)
        
        # Calculate metrics
        test_metrics = self._calculate_metrics(y_test, y_test_pred, y_test_proba)
        
        # Store test results
        model_results['test_metrics'] = test_metrics
        model_results['y_test_pred'] = y_test_pred
        model_results['y_test_proba'] = y_test_proba
        
        logger.info(f"📊 Test Results:")
        logger.info(f"   Accuracy: {test_metrics['accuracy']:.4f}")
        logger.info(f"   F1-score (macro): {test_metrics['f1_macro']:.4f}")
        logger.info(f"   Precision (macro): {test_metrics['precision_macro']:.4f}")
        logger.info(f"   Recall (macro): {test_metrics['recall_macro']:.4f}")
        
        return test_metrics
    
    def plot_model_comparison(self, save_path: str = None):
        """
        Create visualization comparing all trained models.
        
        Args:
            save_path: Path to save the plot (optional)
        """
        if not self.model_results:
            logger.warning("❌ No models trained yet.")
            return
        
        # Prepare data for plotting
        models = []
        train_f1 = []
        val_f1 = []
        train_acc = []
        val_acc = []
        
        for model_name, results in self.model_results.items():
            models.append(results['description'])
            train_f1.append(results['train_metrics']['f1_macro'])
            train_acc.append(results['train_metrics']['accuracy'])
            
            if results['val_metrics']:
                val_f1.append(results['val_metrics']['f1_macro'])
                val_acc.append(results['val_metrics']['accuracy'])
            else:
                val_f1.append(0)
                val_acc.append(0)
        
        # Create comparison plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        x = np.arange(len(models))
        width = 0.35
        
        # F1-score comparison
        ax1.bar(x - width/2, train_f1, width, label='Training', alpha=0.8, color='skyblue')
        ax1.bar(x + width/2, val_f1, width, label='Validation', alpha=0.8, color='lightcoral')
        ax1.set_xlabel('Models')
        ax1.set_ylabel('F1-Score (Macro)')
        ax1.set_title('🎯 Model F1-Score Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy comparison  
        ax2.bar(x - width/2, train_acc, width, label='Training', alpha=0.8, color='lightgreen')
        ax2.bar(x + width/2, val_acc, width, label='Validation', alpha=0.8, color='orange')
        ax2.set_xlabel('Models')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('🎯 Model Accuracy Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels(models, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"💾 Model comparison plot saved to {save_path}")
        
        plt.show()
    
    def plot_confusion_matrix(self, y_true: pd.Series, y_pred: np.ndarray,
                             classes: List[str], title: str = "Confusion Matrix",
                             save_path: str = None):
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            classes: Class names
            title: Plot title
            save_path: Path to save the plot (optional)
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=classes, yticklabels=classes)
        plt.title(f'🎯 {title}', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        # Add percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        for i in range(len(classes)):
            for j in range(len(classes)):
                plt.text(j + 0.5, i + 0.7, f'({cm_percent[i, j]:.1f}%)', 
                        ha='center', va='center', fontsize=10, color='gray')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"💾 Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def plot_classification_report(self, y_true: pd.Series, y_pred: np.ndarray,
                                  classes: List[str], title: str = "Classification Report",
                                  save_path: str = None):
        """
        Plot classification report as heatmap.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            classes: Class names
            title: Plot title
            save_path: Path to save the plot (optional)
        """
        from sklearn.metrics import classification_report
        
        # Get classification report as dictionary
        report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
        
        # Convert to DataFrame for plotting
        df_report = pd.DataFrame(report).iloc[:-1, :-1].T  # Remove 'support' column and summary rows
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(df_report, annot=True, cmap='RdYlBu_r', fmt='.3f', 
                    cbar_kws={'label': 'Score'})
        plt.title(f'📊 {title}', fontsize=16, fontweight='bold')
        plt.xlabel('Metrics')
        plt.ylabel('Classes')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"💾 Classification report saved to {save_path}")
        
        plt.show()
    
    def save_best_model(self, save_dir: str = "models"):
        """
        Save the best trained model.
        
        Args:
            save_dir: Directory to save the model
        """
        if self.best_model is None:
            raise ValueError("No best model selected. Call select_best_model() first.")
        
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        model = self.best_model['model']
        model_name = self.best_model['model_name']
        
        # Save model
        model_file = save_path / f"best_model_{model_name}.joblib"
        joblib.dump(model, model_file)
        
        # Save model metadata
        metadata = {
            'model_name': model_name,
            'description': self.best_model['description'],
            'best_params': self.best_model['best_params'],
            'classes': list(self.best_model['classes']),
            'feature_names': list(self.best_model.get('feature_names', [])),
            'train_metrics': self.best_model['train_metrics'],
            'val_metrics': self.best_model['val_metrics'],
            'test_metrics': self.best_model.get('test_metrics', {}),
            'training_time': self.best_model['training_time'],
            'timestamp': datetime.now().isoformat()
        }
        
        metadata_file = save_path / f"model_metadata_{model_name}.json"
        import json
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save preprocessor if available
        if self.preprocessor:
            preprocessor_file = save_path / "preprocessor.joblib"
            joblib.dump(self.preprocessor, preprocessor_file)
        
        logger.info(f"💾 Best model saved to {save_path}/")
        logger.info(f"   Model: {model_file}")
        logger.info(f"   Metadata: {metadata_file}")

def main():
    """
    Example usage of the model training pipeline.
    """
    print("🚀 NASA Space Apps Challenge 2025 - Model Training")
    print("=" * 60)
    
    # Initialize trainer
    trainer = ExoplanetModelTrainer(random_state=42)
    
    # Load preprocessed data
    print("\n📁 Loading preprocessed data...")
    X, y, sources = trainer.load_preprocessed_data()
    
    if len(X) == 0:
        print("❌ No data available for training.")
        return
    
    # Split data
    print("\n📊 Splitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.split_data(X, y)
    
    # Scale features
    if trainer.preprocessor:
        print("\n⚖️  Scaling features...")
        X_train_scaled, X_val_scaled = trainer.preprocessor.scale_features(X_train, X_val)
        X_test_scaled = trainer.preprocessor.scale_features(X_test)[0] if hasattr(trainer.preprocessor.scale_features(X_test), '__getitem__') else trainer.preprocessor.scale_features(X_test)
    else:
        from preprocess import ExoplanetDataPreprocessor
        temp_preprocessor = ExoplanetDataPreprocessor()
        X_train_scaled, X_val_scaled = temp_preprocessor.scale_features(X_train, X_val)
        X_test_scaled = temp_preprocessor.scaler.transform(X_test)
        trainer.preprocessor = temp_preprocessor
    
    # Train all models
    print("\n🤖 Training models...")
    all_results = trainer.train_all_models(X_train_scaled, y_train, X_val_scaled, y_val)
    
    # Select best model
    print("\n🏆 Selecting best model...")
    best_model = trainer.select_best_model('f1_macro')
    
    # Test best model
    print("\n🧪 Evaluating on test set...")
    test_metrics = trainer.evaluate_on_test(X_test_scaled, y_test)
    
    # Create visualizations
    print("\n📊 Creating visualizations...")
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    # Model comparison
    trainer.plot_model_comparison(save_path=output_dir / "model_comparison.png")
    
    # Confusion matrix for best model
    y_test_pred = best_model['y_test_pred']
    classes = list(best_model['classes'])
    
    trainer.plot_confusion_matrix(
        y_test, y_test_pred, classes, 
        title=f"{best_model['description']} - Test Set",
        save_path=output_dir / "confusion_matrix.png"
    )
    
    # Classification report
    trainer.plot_classification_report(
        y_test, y_test_pred, classes,
        title=f"{best_model['description']} - Classification Report",
        save_path=output_dir / "classification_report.png"
    )
    
    # Save best model
    print("\n💾 Saving best model...")
    trainer.save_best_model()
    
    print("\n✅ Training pipeline complete!")
    print(f"   Best Model: {best_model['description']}")
    print(f"   Test F1-score: {test_metrics['f1_macro']:.4f}")
    print(f"   Test Accuracy: {test_metrics['accuracy']:.4f}")

if __name__ == "__main__":
    main()