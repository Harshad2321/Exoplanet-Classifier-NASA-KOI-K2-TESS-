#!/usr/bin/env python3
"""
🚀 Neural Architecture Search for Exoplanet Classification
Advanced Deep Learning with GPU Optimization
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import time
import json

class NeuralArchitectureSearch:
    """Advanced neural architecture search for exoplanet classification"""
    
    def __init__(self, data_path="data/processed", random_state=42):
        self.data_path = data_path
        self.random_state = random_state
        
        # GPU Configuration
        self._configure_gpu()
        
        # Create directories
        os.makedirs("models/neural_search", exist_ok=True)
        os.makedirs("results/neural_search", exist_ok=True)
        
        print("🧠 Neural Architecture Search for Exoplanet Classification")
        print("=" * 60)
        
    def _configure_gpu(self):
        """Configure GPU for optimal performance"""
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"🔥 Found {len(gpus)} GPU(s), memory growth enabled")
            except RuntimeError as e:
                print(f"⚠️ GPU configuration error: {e}")
        else:
            print("💻 No GPU found, using CPU")
            
    def load_and_prepare_data(self):
        """Load and prepare data with advanced preprocessing"""
        print("📥 Loading and preparing data...")
        
        # Load data
        features = pd.read_csv(f"{self.data_path}/features.csv")
        labels = pd.read_csv(f"{self.data_path}/labels.csv")
        
        print(f"📊 Dataset: {features.shape[0]} samples, {features.shape[1]} features")
        print(f"🏷️ Classes: {labels['label'].value_counts().to_dict()}")
        
        # Advanced feature engineering
        enhanced_features = self._engineer_neural_features(features)
        
        # Encode labels
        le = LabelEncoder()
        y = le.fit_transform(labels['label'])
        self.label_encoder = le
        self.n_classes = len(le.classes_)
        self.class_names = le.classes_
        
        print(f"🔧 Enhanced features: {enhanced_features.shape[1]}")
        print(f"🎯 Classes: {self.n_classes}")
        
        return enhanced_features, y
    
    def _engineer_neural_features(self, df):
        """Advanced feature engineering for neural networks"""
        features = df.copy()
        
        # Mathematical transformations
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            # Multiple transformations
            features[f'{col}_log'] = np.log1p(np.abs(features[col]) + 1e-8)
            features[f'{col}_sqrt'] = np.sqrt(np.abs(features[col]))
            features[f'{col}_square'] = features[col] ** 2
            features[f'{col}_cube'] = features[col] ** 3
            
            # Trigonometric features for periodic patterns
            features[f'{col}_sin'] = np.sin(features[col])
            features[f'{col}_cos'] = np.cos(features[col])
            
        # Physics-inspired features
        if all(col in features.columns for col in ['period', 'radius']):
            # Kepler's third law inspired
            features['kepler_ratio'] = features['period'] ** (2/3) / (features['radius'] + 1e-8)
            features['orbital_velocity'] = 2 * np.pi * features['radius'] / (features['period'] + 1e-8)
            
        if all(col in features.columns for col in ['temperature', 'insolation']):
            # Stefan-Boltzmann inspired
            features['luminosity_proxy'] = features['temperature'] ** 4 * features['insolation']
            features['habitable_zone'] = features['temperature'] / np.sqrt(features['insolation'] + 1e-8)
            
        if all(col in features.columns for col in ['depth', 'radius']):
            # Transit photometry
            features['transit_area'] = np.pi * (features['radius'] ** 2) * features['depth'] / 1e6
            features['relative_size'] = features['radius'] / np.sqrt(features['depth'] + 1e-8)
            
        # Binned categorical features
        if 'temperature' in features.columns:
            # Stellar classification inspired bins
            temp_bins = [0, 3700, 5200, 6000, 7500, 10000, np.inf]
            features['stellar_class'] = pd.cut(features['temperature'], bins=temp_bins, labels=False)
            
        # Remove infinite and NaN values
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(features.median())
        
        return features
        
    def create_architecture_candidates(self, input_dim):
        """Create diverse neural architecture candidates"""
        print("🏗️ Creating neural architecture candidates...")
        
        architectures = {}
        
        # 1. Deep Dense Network
        def deep_dense():
            model = keras.Sequential([
                layers.Dense(512, activation='relu', input_shape=(input_dim,)),
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
                layers.Dense(self.n_classes, activation='softmax')
            ])
            return model
        
        architectures['DeepDense'] = deep_dense
        
        # 2. Wide Network
        def wide_network():
            model = keras.Sequential([
                layers.Dense(1024, activation='relu', input_shape=(input_dim,)),
                layers.BatchNormalization(),
                layers.Dropout(0.4),
                layers.Dense(512, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                layers.Dense(self.n_classes, activation='softmax')
            ])
            return model
        
        architectures['WideNetwork'] = wide_network
        
        # 3. ResNet-inspired with Skip Connections
        def resnet_inspired():
            inputs = layers.Input(shape=(input_dim,))
            
            # Initial dense layer
            x = layers.Dense(256, activation='relu')(inputs)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.3)(x)
            
            # Residual blocks
            for _ in range(3):
                residual = x
                x = layers.Dense(256, activation='relu')(x)
                x = layers.BatchNormalization()(x)
                x = layers.Dropout(0.2)(x)
                x = layers.Dense(256, activation='relu')(x)
                x = layers.BatchNormalization()(x)
                # Skip connection
                x = layers.Add()([x, residual])
                x = layers.Dropout(0.2)(x)
            
            # Output layer
            outputs = layers.Dense(self.n_classes, activation='softmax')(x)
            
            model = keras.Model(inputs=inputs, outputs=outputs)
            return model
        
        architectures['ResNetInspired'] = resnet_inspired
        
        # 4. Attention-based Network
        def attention_network():
            inputs = layers.Input(shape=(input_dim,))
            
            # Feature embedding
            x = layers.Dense(256, activation='relu')(inputs)
            x = layers.BatchNormalization()(x)
            
            # Multi-head attention mechanism (simplified)
            attention_dim = 64
            
            # Query, Key, Value
            query = layers.Dense(attention_dim)(x)
            key = layers.Dense(attention_dim)(x)
            value = layers.Dense(attention_dim)(x)
            
            # Attention weights
            attention_weights = tf.nn.softmax(
                tf.matmul(query, key, transpose_b=True) / np.sqrt(attention_dim)
            )
            
            # Apply attention
            attended = tf.matmul(attention_weights, value)
            
            # Combine with original features
            combined = layers.Concatenate()([x, attended])
            
            # Final layers
            x = layers.Dense(128, activation='relu')(combined)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.3)(x)
            
            outputs = layers.Dense(self.n_classes, activation='softmax')(x)
            
            model = keras.Model(inputs=inputs, outputs=outputs)
            return model
        
        architectures['AttentionNetwork'] = attention_network
        
        # 5. Ensemble of Experts
        def mixture_of_experts():
            inputs = layers.Input(shape=(input_dim,))
            
            # Expert networks
            experts = []
            for i in range(3):
                expert = layers.Dense(128, activation='relu', name=f'expert_{i}_1')(inputs)
                expert = layers.BatchNormalization(name=f'expert_{i}_bn1')(expert)
                expert = layers.Dropout(0.2, name=f'expert_{i}_dropout1')(expert)
                expert = layers.Dense(64, activation='relu', name=f'expert_{i}_2')(expert)
                expert = layers.Dense(32, activation='relu', name=f'expert_{i}_3')(expert)
                experts.append(expert)
            
            # Gating network
            gate = layers.Dense(64, activation='relu')(inputs)
            gate = layers.Dense(3, activation='softmax')(gate)  # 3 experts
            
            # Weighted combination
            expert_outputs = layers.Concatenate()(experts)
            expert_outputs = layers.Reshape((3, 32))(expert_outputs)
            
            # Apply gating
            gate_expanded = layers.RepeatVector(32)(gate)
            gate_expanded = layers.Permute((2, 1))(gate_expanded)
            
            weighted_experts = layers.Multiply()([expert_outputs, gate_expanded])
            combined = tf.reduce_sum(weighted_experts, axis=1)
            
            outputs = layers.Dense(self.n_classes, activation='softmax')(combined)
            
            model = keras.Model(inputs=inputs, outputs=outputs)
            return model
        
        architectures['MixtureOfExperts'] = mixture_of_experts
        
        print(f"📐 Created {len(architectures)} architecture candidates")
        return architectures
        
    def train_and_evaluate_architecture(self, architecture_fn, X_train, y_train, X_val, y_val, name):
        """Train and evaluate a single architecture"""
        print(f"🔧 Training {name}...")
        
        # Create model
        model = architecture_fn()
        
        # Compile with advanced optimizer
        model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=0.001,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-7
            ),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Advanced callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                patience=15,
                restore_best_weights=True,
                monitor='val_accuracy'
            ),
            keras.callbacks.ReduceLROnPlateau(
                factor=0.5,
                patience=8,
                min_lr=1e-7,
                monitor='val_accuracy'
            ),
            keras.callbacks.ModelCheckpoint(
                f'models/neural_search/{name.lower()}_best.keras',
                save_best_only=True,
                monitor='val_accuracy'
            )
        ]
        
        # Train with timing
        start_time = time.time()
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=128,
            callbacks=callbacks,
            verbose=0
        )
        
        training_time = time.time() - start_time
        
        # Evaluate
        val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
        
        # Get best metrics from history
        best_val_accuracy = max(history.history['val_accuracy'])
        best_epoch = history.history['val_accuracy'].index(best_val_accuracy) + 1
        
        results = {
            'model': model,
            'history': history,
            'val_accuracy': val_accuracy,
            'best_val_accuracy': best_val_accuracy,
            'best_epoch': best_epoch,
            'training_time': training_time,
            'total_params': model.count_params()
        }
        
        print(f"   ✅ {name}: {best_val_accuracy:.4f} best accuracy (epoch {best_epoch}, {training_time:.1f}s)")
        
        return results
        
    def neural_architecture_search(self, X_train, y_train, X_val, y_val):
        """Perform neural architecture search"""
        print("🔍 Neural Architecture Search...")
        
        # Get architecture candidates
        architectures = self.create_architecture_candidates(X_train.shape[1])
        
        # Train and evaluate each architecture
        results = {}
        
        for name, architecture_fn in architectures.items():
            try:
                results[name] = self.train_and_evaluate_architecture(
                    architecture_fn, X_train, y_train, X_val, y_val, name
                )
            except Exception as e:
                print(f"   ❌ {name} failed: {str(e)}")
                continue
                
        return results
        
    def evaluate_final_models(self, results, X_test, y_test):
        """Evaluate the best architectures on test set"""
        print("🏆 Final evaluation on test set...")
        
        final_results = {}
        
        for name, result in results.items():
            model = result['model']
            
            # Test predictions
            y_pred_proba = model.predict(X_test, verbose=0)
            y_pred = np.argmax(y_pred_proba, axis=1)
            
            # Calculate metrics
            test_accuracy = accuracy_score(y_test, y_pred)
            
            final_results[name] = {
                'test_accuracy': test_accuracy,
                'val_accuracy': result['best_val_accuracy'],
                'training_time': result['training_time'],
                'total_params': result['total_params'],
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"📊 {name:<20} Test: {test_accuracy:.4f}  Val: {result['best_val_accuracy']:.4f}  Params: {result['total_params']:,}")
            
        return final_results
        
    def create_ensemble_prediction(self, results, X_test):
        """Create ensemble prediction from top models"""
        print("🤝 Creating neural ensemble...")
        
        # Sort by validation accuracy
        sorted_results = sorted(results.items(), key=lambda x: x[1]['val_accuracy'], reverse=True)
        
        # Select top 3 models for ensemble
        top_models = sorted_results[:3]
        
        print(f"📊 Ensemble from top {len(top_models)} models:")
        for name, result in top_models:
            print(f"   - {name}: {result['val_accuracy']:.4f}")
        
        # Get predictions from each model
        ensemble_predictions = []
        
        for name, result in top_models:
            model = result['model']
            pred_proba = model.predict(X_test, verbose=0)
            ensemble_predictions.append(pred_proba)
        
        # Average predictions
        ensemble_proba = np.mean(ensemble_predictions, axis=0)
        ensemble_pred = np.argmax(ensemble_proba, axis=1)
        
        return ensemble_pred, ensemble_proba
        
    def generate_comprehensive_report(self, final_results, ensemble_pred, ensemble_proba, y_test):
        """Generate comprehensive analysis report"""
        print("📋 Generating comprehensive report...")
        
        # Sort results by test accuracy
        sorted_results = sorted(final_results.items(), key=lambda x: x[1]['test_accuracy'], reverse=True)
        
        # Best individual model
        best_name, best_result = sorted_results[0]
        
        # Ensemble accuracy
        ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
        
        # Create results summary
        results_summary = []
        for name, result in sorted_results:
            results_summary.append({
                'Model': name,
                'Test_Accuracy': result['test_accuracy'],
                'Val_Accuracy': result['val_accuracy'],
                'Parameters': result['total_params'],
                'Training_Time': result['training_time']
            })
        
        # Add ensemble
        results_summary.append({
            'Model': 'Neural_Ensemble',
            'Test_Accuracy': ensemble_accuracy,
            'Val_Accuracy': np.mean([r['val_accuracy'] for r in final_results.values()]),
            'Parameters': sum([r['total_params'] for r in final_results.values()]),
            'Training_Time': sum([r['training_time'] for r in final_results.values()])
        })
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_df = pd.DataFrame(results_summary)
        results_df = results_df.sort_values('Test_Accuracy', ascending=False)
        
        results_path = f"results/neural_search/neural_search_results_{timestamp}.csv"
        results_df.to_csv(results_path, index=False)
        
        # Classification report for ensemble
        class_report = classification_report(
            y_test, ensemble_pred,
            target_names=self.class_names,
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_test, ensemble_pred)
        
        # Visualizations
        self._create_visualizations(results_df, cm, class_report, timestamp)
        
        # Print summary
        print("\n🧠 NEURAL ARCHITECTURE SEARCH RESULTS")
        print("=" * 80)
        print(f"{'Model':<20} {'Test Acc':<10} {'Val Acc':<10} {'Params':<12} {'Time':<10}")
        print("-" * 80)
        
        for _, row in results_df.iterrows():
            params_str = f"{row['Parameters']:,}" if row['Parameters'] < 1e6 else f"{row['Parameters']/1e6:.1f}M"
            print(f"{row['Model']:<20} {row['Test_Accuracy']:<10.4f} {row['Val_Accuracy']:<10.4f} "
                  f"{params_str:<12} {row['Training_Time']:<10.1f}s")
        
        print("-" * 80)
        best_model = results_df.iloc[0]['Model']
        best_accuracy = results_df.iloc[0]['Test_Accuracy']
        
        print(f"🏅 Best Model: {best_model}")
        print(f"🎯 Best Accuracy: {best_accuracy:.4f}")
        print(f"📈 Improvement over baseline: +{(best_accuracy - 0.701)*100:.2f}%")
        print(f"📁 Results saved to: {results_path}")
        
        return results_df
        
    def _create_visualizations(self, results_df, cm, class_report, timestamp):
        """Create comprehensive visualizations"""
        
        # 1. Model comparison
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        models = results_df['Model'].head(6)  # Top 6 models
        accuracies = results_df['Test_Accuracy'].head(6)
        
        bars = plt.bar(range(len(models)), accuracies)
        plt.xticks(range(len(models)), models, rotation=45, ha='right')
        plt.ylabel('Test Accuracy')
        plt.title('Neural Architecture Comparison')
        plt.ylim(min(accuracies) * 0.95, max(accuracies) * 1.02)
        
        # Color bars by performance
        for i, bar in enumerate(bars):
            bar.set_color(plt.cm.viridis(accuracies.iloc[i]))
        
        # 2. Parameters vs Accuracy
        plt.subplot(2, 2, 2)
        params = results_df['Parameters'].head(6)
        plt.scatter(params, accuracies, s=100, alpha=0.7, c=accuracies, cmap='viridis')
        plt.xlabel('Model Parameters')
        plt.ylabel('Test Accuracy')
        plt.title('Parameters vs Accuracy')
        plt.xscale('log')
        
        # 3. Confusion Matrix
        plt.subplot(2, 2, 3)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Confusion Matrix (Best Ensemble)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # 4. Per-class F1 scores
        plt.subplot(2, 2, 4)
        f1_scores = [class_report[class_name]['f1-score'] for class_name in self.class_names]
        bars = plt.bar(self.class_names, f1_scores, color='skyblue')
        plt.ylabel('F1 Score')
        plt.title('Per-class F1 Scores')
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar, score in zip(bars, f1_scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'results/neural_search/neural_analysis_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("📊 Visualizations saved to results/neural_search/")
        
    def run_neural_search(self):
        """Run the complete neural architecture search"""
        print("\n🧠 Starting Neural Architecture Search")
        print("=" * 60)
        
        # 1. Load and prepare data
        X, y = self.load_and_prepare_data()
        
        # 2. Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=self.random_state
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=self.random_state
        )
        
        print(f"📊 Data splits - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # 3. Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # 4. Neural architecture search
        search_results = self.neural_architecture_search(X_train_scaled, y_train, X_val_scaled, y_val)
        
        if not search_results:
            print("❌ No architectures completed successfully")
            return None
        
        # 5. Final evaluation
        final_results = self.evaluate_final_models(search_results, X_test_scaled, y_test)
        
        # 6. Create ensemble
        ensemble_pred, ensemble_proba = self.create_ensemble_prediction(final_results, X_test_scaled)
        
        # 7. Generate comprehensive report
        results_df = self.generate_comprehensive_report(final_results, ensemble_pred, ensemble_proba, y_test)
        
        print("\n✅ Neural Architecture Search Complete!")
        return results_df

def main():
    """Run neural architecture search"""
    searcher = NeuralArchitectureSearch()
    results = searcher.run_neural_search()
    return results

if __name__ == "__main__":
    results = main()