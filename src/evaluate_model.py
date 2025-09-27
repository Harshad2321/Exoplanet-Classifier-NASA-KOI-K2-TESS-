"""
Model Evaluation Module for NASA Exoplanet Classification

This module handles:
- Comprehensive model evaluation and metrics
- Visualization of results (confusion matrices, ROC curves, feature importance)
- SHAP/LIME explanations for model interpretability
- Performance analysis and reporting

Author: NASA Space Apps Challenge 2025 Team
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import logging
import joblib
import warnings

# ML evaluation
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score, roc_auc_score
)

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Explainability
import shap
import lime
from lime.lime_tabular import LimeTabularExplainer

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExoplanetModelEvaluator:
    """
    Comprehensive model evaluation and visualization for exoplanet classification
    """
    
    def __init__(self, models_dir: str = "models", data_dir: str = "data"):
        """
        Initialize model evaluator
        
        Args:
            models_dir: Directory containing trained models
            data_dir: Directory containing processed data
        """
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)
        self.splits_dir = self.data_dir / "splits"
        self.processed_dir = self.data_dir / "processed"
        
        # Initialize storage
        self.models = {}
        self.model_scores = {}
        self.feature_names = None
        self.target_mapping = None
        
        # Test data
        self.X_test = None
        self.y_test = None
        
        # Predictions storage
        self.predictions = {}
        self.probabilities = {}
    
    def load_models_and_data(self) -> bool:
        """
        Load trained models and test data
        
        Returns:
            Success status
        """
        try:
            # Load test data
            test_df = pd.read_csv(self.splits_dir / 'test.csv')
            self.feature_names = joblib.load(self.processed_dir / 'feature_names.pkl')
            self.target_mapping = joblib.load(self.processed_dir / 'target_mapping.pkl')
            
            self.X_test = test_df[self.feature_names]
            self.y_test = test_df.drop(columns=self.feature_names).iloc[:, 0]
            
            # Load models
            model_files = list(self.models_dir.glob("*_model.pkl"))
            if (self.models_dir / "best_model.pkl").exists():
                model_files.append(self.models_dir / "best_model.pkl")
            
            for model_file in model_files:
                model_name = model_file.stem.replace('_model', '')
                if model_name == 'best':
                    # Get best model name from metadata
                    try:
                        metadata = joblib.load(self.models_dir / 'model_metadata.pkl')
                        model_name = f"best_{metadata.get('best_model', 'unknown')}"
                    except:
                        model_name = 'best_model'
                
                self.models[model_name] = joblib.load(model_file)
                logger.info(f"Loaded {model_name}")
            
            # Load model scores if available
            if (self.models_dir / 'model_scores.pkl').exists():
                self.model_scores = joblib.load(self.models_dir / 'model_scores.pkl')
            
            logger.info(f"✅ Loaded {len(self.models)} models and test data ({len(self.X_test)} samples)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load models and data: {e}")
            return False
    
    def evaluate_all_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate all loaded models on test data
        
        Returns:
            Comprehensive evaluation results for all models
        """
        logger.info("🔍 Evaluating all models...")
        
        evaluation_results = {}
        
        for model_name, model in self.models.items():
            logger.info(f"Evaluating {model_name}...")
            
            # Make predictions
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)
            
            # Store predictions
            self.predictions[model_name] = y_pred
            self.probabilities[model_name] = y_pred_proba
            
            # Calculate metrics
            metrics = self._calculate_comprehensive_metrics(self.y_test, y_pred, y_pred_proba)
            
            # Generate classification report
            class_names = [k for k, v in sorted(self.target_mapping.items(), key=lambda x: x[1])]
            report = classification_report(
                self.y_test, y_pred, 
                target_names=class_names,
                output_dict=True
            )
            
            evaluation_results[model_name] = {
                'metrics': metrics,
                'classification_report': report,
                'confusion_matrix': confusion_matrix(self.y_test, y_pred),
                'class_names': class_names
            }
        
        logger.info("✅ Model evaluation completed!")
        return evaluation_results
    
    def _calculate_comprehensive_metrics(self, y_true, y_pred, y_pred_proba) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics"""
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'precision_micro': precision_score(y_true, y_pred, average='micro', zero_division=0),
            'recall_micro': recall_score(y_true, y_pred, average='micro', zero_division=0),
            'f1_micro': f1_score(y_true, y_pred, average='micro', zero_division=0)
        }
        
        # Add per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        class_names = [k for k, v in sorted(self.target_mapping.items(), key=lambda x: x[1])]
        for i, class_name in enumerate(class_names):
            metrics[f'precision_{class_name.lower()}'] = precision_per_class[i]
            metrics[f'recall_{class_name.lower()}'] = recall_per_class[i]
            metrics[f'f1_{class_name.lower()}'] = f1_per_class[i]
        
        # ROC-AUC
        try:
            if len(np.unique(y_true)) > 2:
                metrics['roc_auc_ovr'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
                metrics['roc_auc_ovo'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovo', average='weighted')
            else:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
        except:
            metrics['roc_auc_ovr'] = 0.0
            metrics['roc_auc_ovo'] = 0.0
        
        return metrics
    
    def plot_confusion_matrices(self, evaluation_results: Dict, figsize: Tuple[int, int] = (15, 10)):
        """
        Plot confusion matrices for all models
        
        Args:
            evaluation_results: Results from evaluate_all_models()
            figsize: Figure size
        """
        n_models = len(evaluation_results)
        cols = min(3, n_models)
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if n_models == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, (model_name, results) in enumerate(evaluation_results.items()):
            cm = results['confusion_matrix']
            class_names = results['class_names']
            
            # Create confusion matrix heatmap
            sns.heatmap(
                cm, 
                annot=True, 
                fmt='d',
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                ax=axes[i]
            )
            
            axes[i].set_title(f'{model_name.replace("_", " ").title()}\nAccuracy: {results["metrics"]["accuracy"]:.3f}')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('True')
        
        # Hide empty subplots
        for i in range(n_models, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Save plot
        plt.savefig(self.models_dir / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
        logger.info("Confusion matrices saved to confusion_matrices.png")
    
    def plot_roc_curves(self, evaluation_results: Dict, figsize: Tuple[int, int] = (12, 8)):
        """
        Plot ROC curves for all models
        
        Args:
            evaluation_results: Results from evaluate_all_models()
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(evaluation_results)))
        
        for (model_name, results), color in zip(evaluation_results.items(), colors):
            y_pred_proba = self.probabilities[model_name]
            
            # Handle multiclass ROC
            if len(np.unique(self.y_test)) > 2:
                # Plot ROC for each class
                for i, class_name in enumerate(results['class_names']):
                    # One-vs-rest approach
                    y_true_binary = (self.y_test == i).astype(int)
                    fpr, tpr, _ = roc_curve(y_true_binary, y_pred_proba[:, i])
                    roc_auc = auc(fpr, tpr)
                    
                    plt.plot(
                        fpr, tpr, 
                        color=color, 
                        linestyle='-' if i == 0 else '--' if i == 1 else ':',
                        label=f'{model_name} - {class_name} (AUC = {roc_auc:.2f})',
                        alpha=0.8
                    )
            else:
                # Binary classification
                fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba[:, 1])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, color=color, label=f'{model_name} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Model Comparison')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Save plot
        plt.savefig(self.models_dir / 'roc_curves.png', dpi=300, bbox_inches='tight')
        logger.info("ROC curves saved to roc_curves.png")
    
    def plot_feature_importance(self, model_name: str = None, top_features: int = 20):
        """
        Plot feature importance for tree-based models
        
        Args:
            model_name: Name of model (if None, uses first available tree-based model)
            top_features: Number of top features to show
        """
        # Find a tree-based model if none specified
        if model_name is None:
            tree_models = ['random_forest', 'xgboost', 'lightgbm']
            for name in tree_models:
                if name in self.models:
                    model_name = name
                    break
        
        if model_name not in self.models:
            logger.warning(f"Model {model_name} not found or doesn't support feature importance")
            return
        
        model = self.models[model_name]
        
        # Get feature importance
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_).mean(axis=0)
        else:
            logger.warning(f"Model {model_name} doesn't support feature importance")
            return
        
        # Create DataFrame for plotting
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False).head(top_features)
        
        # Plot
        plt.figure(figsize=(10, 8))
        sns.barplot(data=importance_df, x='importance', y='feature', palette='viridis')
        plt.title(f'Feature Importance - {model_name.replace("_", " ").title()}')
        plt.xlabel('Importance')
        plt.ylabel('Features')
        plt.tight_layout()
        plt.show()
        
        # Save plot
        plt.savefig(self.models_dir / f'feature_importance_{model_name}.png', dpi=300, bbox_inches='tight')
        logger.info(f"Feature importance plot saved for {model_name}")
    
    def explain_predictions_shap(self, model_name: str, n_samples: int = 100):
        """
        Generate SHAP explanations for model predictions
        
        Args:
            model_name: Name of model to explain
            n_samples: Number of samples for background dataset
        """
        if model_name not in self.models:
            logger.error(f"Model {model_name} not found")
            return
        
        try:
            model = self.models[model_name]
            
            # Create explainer
            # Use a sample of training data as background
            X_background = self.X_test.sample(n=min(n_samples, len(self.X_test)), random_state=42)
            
            if hasattr(model, 'predict_proba'):
                explainer = shap.KernelExplainer(model.predict_proba, X_background)
            else:
                explainer = shap.KernelExplainer(model.predict, X_background)
            
            # Calculate SHAP values for a subset of test data
            X_explain = self.X_test.sample(n=min(20, len(self.X_test)), random_state=42)
            shap_values = explainer.shap_values(X_explain)
            
            # Plot SHAP summary
            plt.figure(figsize=(10, 6))
            if isinstance(shap_values, list):  # Multi-class
                shap.summary_plot(shap_values[0], X_explain, feature_names=self.feature_names, show=False)
            else:
                shap.summary_plot(shap_values, X_explain, feature_names=self.feature_names, show=False)
            
            plt.title(f'SHAP Summary - {model_name.replace("_", " ").title()}')
            plt.tight_layout()
            plt.show()
            
            # Save plot
            plt.savefig(self.models_dir / f'shap_summary_{model_name}.png', dpi=300, bbox_inches='tight')
            logger.info(f"SHAP summary saved for {model_name}")
            
            return shap_values
            
        except Exception as e:
            logger.error(f"Failed to generate SHAP explanations: {e}")
            return None
    
    def explain_predictions_lime(self, model_name: str, instance_idx: int = 0):
        """
        Generate LIME explanations for a specific prediction
        
        Args:
            model_name: Name of model to explain
            instance_idx: Index of test instance to explain
        """
        if model_name not in self.models:
            logger.error(f"Model {model_name} not found")
            return
        
        try:
            model = self.models[model_name]
            
            # Create LIME explainer
            explainer = LimeTabularExplainer(
                self.X_test.values,
                feature_names=self.feature_names,
                class_names=[k for k, v in sorted(self.target_mapping.items(), key=lambda x: x[1])],
                mode='classification'
            )
            
            # Explain instance
            instance = self.X_test.iloc[instance_idx].values
            explanation = explainer.explain_instance(instance, model.predict_proba, num_features=10)
            
            # Save explanation
            explanation.save_to_file(str(self.models_dir / f'lime_explanation_{model_name}_{instance_idx}.html'))
            logger.info(f"LIME explanation saved for {model_name}, instance {instance_idx}")
            
            return explanation
            
        except Exception as e:
            logger.error(f"Failed to generate LIME explanation: {e}")
            return None
    
    def generate_evaluation_report(self, evaluation_results: Dict) -> pd.DataFrame:
        """
        Generate comprehensive evaluation report
        
        Args:
            evaluation_results: Results from evaluate_all_models()
            
        Returns:
            DataFrame with detailed evaluation metrics
        """
        report_data = []
        
        for model_name, results in evaluation_results.items():
            metrics = results['metrics']
            
            report_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision (Macro)': f"{metrics['precision_macro']:.4f}",
                'Recall (Macro)': f"{metrics['recall_macro']:.4f}",
                'F1-Score (Macro)': f"{metrics['f1_macro']:.4f}",
                'Precision (Weighted)': f"{metrics['precision_weighted']:.4f}",
                'Recall (Weighted)': f"{metrics['recall_weighted']:.4f}",
                'F1-Score (Weighted)': f"{metrics['f1_weighted']:.4f}",
                'ROC-AUC (OvR)': f"{metrics.get('roc_auc_ovr', 0):.4f}"
            })
        
        report_df = pd.DataFrame(report_data)
        
        # Sort by F1-Score (Weighted)
        report_df['F1_numeric'] = report_df['F1-Score (Weighted)'].astype(float)
        report_df = report_df.sort_values('F1_numeric', ascending=False)
        report_df = report_df.drop('F1_numeric', axis=1)
        
        # Save report
        report_df.to_csv(self.models_dir / 'evaluation_report.csv', index=False)
        logger.info("Evaluation report saved to evaluation_report.csv")
        
        return report_df
    
    def create_interactive_dashboard(self, evaluation_results: Dict):
        """
        Create interactive Plotly dashboard for model evaluation
        
        Args:
            evaluation_results: Results from evaluate_all_models()
        """
        try:
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Model Accuracy Comparison', 'F1-Score Comparison', 
                              'Precision-Recall', 'Class Distribution'),
                specs=[[{"type": "bar"}, {"type": "bar"}],
                       [{"type": "scatter"}, {"type": "pie"}]]
            )
            
            # Extract model names and metrics
            model_names = []
            accuracies = []
            f1_scores = []
            
            for model_name, results in evaluation_results.items():
                model_names.append(model_name.replace('_', ' ').title())
                accuracies.append(results['metrics']['accuracy'])
                f1_scores.append(results['metrics']['f1_weighted'])
            
            # Add bar charts
            fig.add_trace(
                go.Bar(x=model_names, y=accuracies, name='Accuracy', marker_color='lightblue'),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Bar(x=model_names, y=f1_scores, name='F1-Score', marker_color='lightcoral'),
                row=1, col=2
            )
            
            # Add precision-recall scatter
            precisions = [results['metrics']['precision_weighted'] for results in evaluation_results.values()]
            recalls = [results['metrics']['recall_weighted'] for results in evaluation_results.values()]
            
            fig.add_trace(
                go.Scatter(
                    x=recalls, y=precisions, mode='markers+text',
                    text=model_names, textposition='top center',
                    marker=dict(size=10, color='green'),
                    name='Precision vs Recall'
                ),
                row=2, col=1
            )
            
            # Add class distribution pie chart
            class_counts = pd.Series(self.y_test).value_counts()
            class_names = [k for k, v in sorted(self.target_mapping.items(), key=lambda x: x[1])]
            
            fig.add_trace(
                go.Pie(
                    labels=class_names,
                    values=[class_counts.get(i, 0) for i in range(len(class_names))],
                    name='Class Distribution'
                ),
                row=2, col=2
            )
            
            # Update layout
            fig.update_layout(
                height=800,
                title_text="Exoplanet Classification Model Evaluation Dashboard",
                showlegend=False
            )
            
            # Save interactive plot
            fig.write_html(str(self.models_dir / 'evaluation_dashboard.html'))
            logger.info("Interactive dashboard saved to evaluation_dashboard.html")
            
            return fig
            
        except Exception as e:
            logger.error(f"Failed to create interactive dashboard: {e}")
            return None


def main():
    """
    Main function to demonstrate evaluation functionality
    """
    print("📊 NASA Exoplanet Model Evaluation")
    print("=" * 50)
    
    evaluator = ExoplanetModelEvaluator()
    
    # Load models and data
    if not evaluator.load_models_and_data():
        print("❌ Failed to load models and data. Train models first.")
        return
    
    # Evaluate all models
    print("\n🔍 Evaluating models...")
    evaluation_results = evaluator.evaluate_all_models()
    
    # Generate report
    print("\n📋 Evaluation Report:")
    report_df = evaluator.generate_evaluation_report(evaluation_results)
    print(report_df.to_string(index=False))
    
    # Create visualizations
    print("\n📈 Creating visualizations...")
    evaluator.plot_confusion_matrices(evaluation_results)
    evaluator.plot_roc_curves(evaluation_results)
    evaluator.plot_feature_importance()
    
    # Create interactive dashboard
    evaluator.create_interactive_dashboard(evaluation_results)
    
    print("\n✅ Evaluation complete! Check the models directory for saved plots and reports.")


if __name__ == "__main__":
    main()