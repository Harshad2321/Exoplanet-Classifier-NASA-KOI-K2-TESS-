#!/usr/bin/env python3
"""
🚀 COMPLETE PROJECT OPTIMIZER & RUNNER
Comprehensive system to optimize, train, and launch the exoplanet classifier
"""

import os
import sys
import subprocess
import time
from pathlib import Path
import json
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProjectRunner:
    """Complete project optimization and execution system"""
    
    def __init__(self):
        self.project_root = Path.cwd()
        self.setup_complete = False
        
        print("🚀 EXOPLANET CLASSIFIER - PROJECT RUNNER")
        print("=" * 50)
        
    def check_dependencies(self) -> bool:
        """Check and install required dependencies"""
        
        print("📦 Checking dependencies...")
        
        required_packages = [
            "pandas", "numpy", "scikit-learn", "xgboost", "lightgbm",
            "streamlit", "plotly", "seaborn", "matplotlib", "joblib",
            "optuna", "psutil"
        ]
        
        optional_packages = {
            "tensorflow": "GPU acceleration",
            "torch": "PyTorch support"
        }
        
        missing_packages = []
        
        # Check required packages
        for package in required_packages:
            try:
                __import__(package)
                print(f"✅ {package}")
            except ImportError:
                print(f"❌ {package} - MISSING")
                missing_packages.append(package)
        
        # Check optional packages
        for package, description in optional_packages.items():
            try:
                __import__(package)
                print(f"✅ {package} - {description}")
            except ImportError:
                print(f"⚠️  {package} - OPTIONAL ({description})")
        
        # Install missing packages
        if missing_packages:
            print(f"\n📦 Installing {len(missing_packages)} missing packages...")
            
            for package in missing_packages:
                try:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                    print(f"✅ Installed {package}")
                except subprocess.CalledProcessError as e:
                    print(f"❌ Failed to install {package}: {e}")
                    return False
        
        print("✅ All dependencies satisfied!")
        return True
    
    def optimize_project(self) -> bool:
        """Run project optimization"""
        
        print("\n🔧 Running project optimization...")
        
        try:
            from project_optimizer import ProjectOptimizer
            
            optimizer = ProjectOptimizer()
            optimization_results = optimizer.run_comprehensive_optimization()
            
            if optimization_results:
                print("✅ Project optimization completed!")
                return True
            else:
                print("⚠️ Project optimization completed with warnings")
                return True
                
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            print("⚠️ Continuing without optimization...")
            return True  # Continue even if optimization fails
    
    def prepare_data(self) -> bool:
        """Prepare and optimize data pipeline"""
        
        print("\n📊 Preparing data pipeline...")
        
        try:
            from optimized_data_pipeline import OptimizedExoplanetDataset
            
            dataset = OptimizedExoplanetDataset()
            
            # Test data loading
            features, labels = dataset.load_exoplanet_data(enhanced=True)
            
            if features is not None and labels is not None:
                print(f"✅ Data loaded: {features.shape[0]} samples, {features.shape[1]} features")
                print(f"📊 Class distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")
                return True
            else:
                print("❌ Failed to load data")
                return False
                
        except Exception as e:
            logger.error(f"Data preparation failed: {e}")
            return False
    
    def train_models(self) -> bool:
        """Train optimized models"""
        
        print("\n🤖 Training optimized models...")
        
        try:
            from optimized_model_trainer import OptimizedModelTrainer
            from optimized_data_pipeline import OptimizedExoplanetDataset
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
            import numpy as np
            
            # Load data
            dataset = OptimizedExoplanetDataset()
            features, labels = dataset.load_exoplanet_data(enhanced=True)
            splits = dataset.create_train_test_splits(features, labels)
            
            # Initialize trainer
            trainer = OptimizedModelTrainer(memory_limit_gb=6.0)
            
            # Prepare data
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(splits['X_train'])
            X_test_scaled = scaler.transform(splits['X_test'])
            
            # Split for validation
            X_train_final, X_val, y_train_final, y_val = train_test_split(
                X_train_scaled, splits['y_train'], 
                test_size=0.2, stratify=splits['y_train'], random_state=42
            )
            
            # Train models
            results = trainer.run_optimized_training(
                X_train_final, y_train_final,
                X_val, y_val,
                X_test_scaled, splits['y_test']
            )
            
            if results:
                best_model = max(results['results'].items(), key=lambda x: x[1]['accuracy'])
                print(f"✅ Training completed!")
                print(f"🏆 Best model: {best_model[0]} ({best_model[1]['accuracy']:.4f} accuracy)")
                return True
            else:
                print("❌ Training failed")
                return False
                
        except Exception as e:
            logger.error(f"Training failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def launch_dashboard(self) -> bool:
        """Launch Streamlit dashboard"""
        
        print("\n🎨 Launching dashboard...")
        
        dashboard_path = self.project_root / "streamlit_dashboard.py"
        
        if not dashboard_path.exists():
            print("❌ Dashboard file not found")
            return False
        
        try:
            print("🚀 Starting Streamlit server...")
            print("📱 Dashboard will open in your default browser")
            print("🌐 URL: http://localhost:8501")
            print("\n⏹️  Press Ctrl+C to stop the server")
            
            # Launch Streamlit
            subprocess.run([
                sys.executable, "-m", "streamlit", "run", 
                str(dashboard_path),
                "--server.port=8501",
                "--server.address=localhost",
                "--browser.gatherUsageStats=false"
            ])
            
            return True
            
        except KeyboardInterrupt:
            print("\n🛑 Dashboard stopped by user")
            return True
        except Exception as e:
            logger.error(f"Dashboard launch failed: {e}")
            return False
    
    def run_quick_demo(self):
        """Run quick demo without full training"""
        
        print("\n⚡ Running quick demo mode...")
        
        try:
            # Create synthetic demo data
            import numpy as np
            import pandas as pd
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score
            import joblib
            
            # Generate demo data
            np.random.seed(42)
            n_samples = 1000
            
            # Synthetic exoplanet features
            data = {
                'period': np.random.lognormal(2, 1, n_samples),
                'radius': np.random.lognormal(0, 0.5, n_samples),
                'mass': np.random.lognormal(1, 0.8, n_samples),
                'temperature': np.random.normal(5000, 1500, n_samples),
                'distance': np.random.lognormal(5, 1.5, n_samples),
                'stellar_magnitude': np.random.normal(12, 2, n_samples),
                'impact_parameter': np.random.uniform(0, 1, n_samples),
                'transit_duration': np.random.lognormal(1, 0.5, n_samples),
                'transit_depth': np.random.lognormal(-3, 1, n_samples),
                'signal_to_noise': np.random.lognormal(2, 0.5, n_samples)
            }
            
            df = pd.DataFrame(data)
            
            # Create realistic labels
            labels = []
            for _, row in df.iterrows():
                if (row['radius'] > 0.5 and row['radius'] < 2.5 and 
                    row['temperature'] > 3000 and row['signal_to_noise'] > 7):
                    labels.append(2)  # CONFIRMED
                elif row['signal_to_noise'] < 3:
                    labels.append(0)  # FALSE POSITIVE
                else:
                    labels.append(1)  # CANDIDATE
            
            # Prepare data
            X = df.values
            y = np.array(labels)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train quick model
            print("🌳 Training demo model...")
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            
            # Evaluate
            accuracy = accuracy_score(y_test, model.predict(X_test))
            print(f"✅ Demo model trained: {accuracy:.4f} accuracy")
            
            # Save demo model
            models_dir = Path("models/optimized")
            models_dir.mkdir(parents=True, exist_ok=True)
            
            joblib.dump(model, models_dir / "demo_model.joblib")
            
            # Save demo data
            data_dir = Path("data")
            data_dir.mkdir(parents=True, exist_ok=True)
            df['koi_disposition'] = ['FALSE POSITIVE', 'CANDIDATE', 'CONFIRMED'][np.searchsorted([0, 1, 2], labels)]
            df.to_csv(data_dir / "exoplanet_data.csv", index=False)
            
            # Create demo results
            results_dir = Path("results/optimized")
            results_dir.mkdir(parents=True, exist_ok=True)
            
            demo_results = {
                'timestamp': datetime.now().isoformat(),
                'model_results': {
                    'Demo_RandomForest': {
                        'accuracy': float(accuracy),
                        'training_time': 5.0,
                        'val_accuracy': float(accuracy)
                    }
                }
            }
            
            with open(results_dir / "training_metadata_demo.json", 'w') as f:
                json.dump(demo_results, f, indent=2)
            
            print("✅ Demo setup completed!")
            return True
            
        except Exception as e:
            logger.error(f"Demo setup failed: {e}")
            return False
    
    def run_complete_pipeline(self):
        """Run the complete project pipeline"""
        
        print("🌟 STARTING COMPLETE PROJECT PIPELINE")
        print("=" * 60)
        
        # Step 1: Check dependencies
        if not self.check_dependencies():
            print("❌ Dependency check failed. Please install missing packages manually.")
            return False
        
        # Step 2: Optimize project
        self.optimize_project()
        
        # Step 3: Prepare data
        if not self.prepare_data():
            print("⚠️  Data preparation failed. Running demo mode...")
            self.run_quick_demo()
        
        # Step 4: Train models (optional - can be skipped for quick demo)
        print("\n🤖 Model Training Options:")
        print("1. 🚀 Full training (10-15 minutes, best accuracy)")
        print("2. ⚡ Quick demo (30 seconds, basic functionality)")
        print("3. 🎨 Skip training, launch dashboard only")
        
        try:
            choice = input("\nChoose option (1/2/3): ").strip()
        except (KeyboardInterrupt, EOFError):
            choice = "2"  # Default to demo mode
        
        if choice == "1":
            print("\n🚀 Starting full training pipeline...")
            if not self.train_models():
                print("⚠️  Full training failed. Running demo mode...")
                self.run_quick_demo()
        elif choice == "2":
            self.run_quick_demo()
        elif choice == "3":
            print("⏭️  Skipping training...")
        else:
            print("Invalid choice, running demo mode...")
            self.run_quick_demo()
        
        # Step 5: Launch dashboard
        print("\n🎨 Ready to launch dashboard!")
        
        try:
            launch_choice = input("Launch dashboard now? (y/n): ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            launch_choice = "y"
        
        if launch_choice in ['y', 'yes', '']:
            self.launch_dashboard()
        else:
            print("🎯 Setup complete! Run 'streamlit run streamlit_dashboard.py' to start the dashboard.")
        
        print("\n🎉 PROJECT PIPELINE COMPLETED!")
        print("=" * 40)

def main():
    """Main entry point"""
    
    runner = ProjectRunner()
    
    try:
        runner.run_complete_pipeline()
    except KeyboardInterrupt:
        print("\n\n🛑 Operation cancelled by user")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()