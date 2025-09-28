# 🪐 Advanced Exoplanet Classification System - NASA Space Apps Challenge 2025

## 🚀 Project Overview

**Enterprise-grade machine learning system** for classifying exoplanets using NASA's comprehensive datasets from Kepler Objects of Interest (KOI), K2, and TESS missions.

> **"Next-Level AI: Advanced Ensemble Learning for Exoplanet Discovery"**

### 🏆 Performance Highlights
- **Multi-Algorithm Ensemble**: 6 advanced ML models with voting classifier  
- **Hyperparameter Optimization**: Optuna-powered automated tuning (100+ trials per model)
- **Production-Ready**: Modular, scalable, and fully documented codebase
- **Interactive Analytics**: Advanced EDA with PCA, t-SNE, and Plotly visualizations
- **Uncertainty Estimation**: Confidence scoring and prediction reliability metrics
- **Comprehensive Evaluation**: Learning curves, feature importance, and model comparison

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![NASA](https://img.shields.io/badge/NASA-Space%20Apps%202025-red.svg)
![Advanced](https://img.shields.io/badge/ML-Advanced%20Ensemble-brightgreen.svg)

---

## 🎯 Advanced Features

### 🤖 Machine Learning Pipeline
- **6 Advanced Algorithms**: Random Forest, Gradient Boosting, XGBoost, LightGBM, SVM, Neural Network
- **Ensemble Methods**: Soft/Hard voting classifiers for improved accuracy  
- **Hyperparameter Tuning**: Optuna optimization with Bayesian optimization
- **Cross-Validation**: Stratified 5-fold validation for robust evaluation
- **Feature Engineering**: Advanced preprocessing with scaling and encoding
- **Model Persistence**: Automated saving/loading with metadata tracking

### 📊 Advanced Analytics & Visualization
- **Dimensionality Reduction**: PCA and t-SNE visualizations for feature space analysis
- **Interactive Dashboards**: Plotly-based visualizations and exploration tools
- **Correlation Analysis**: Advanced feature relationship mapping with heatmaps
- **Statistical Analysis**: Comprehensive data distribution and outlier analysis  
- **Learning Curves**: Training performance analysis and overfitting detection
- **Feature Importance**: Model-agnostic feature contribution analysis

### 🎛️ Production Features
- **Modular Architecture**: Clean, maintainable, and extensible codebase
- **Comprehensive Logging**: Detailed execution tracking and debugging support
- **Batch Prediction**: Scalable inference pipeline with uncertainty estimation
- **Performance Monitoring**: Automated model evaluation and comparison
- **Error Handling**: Robust exception handling and graceful degradation
- **Documentation**: Extensive docstrings and inline documentation

---

## 📊 Dataset Information

### 🌌 Data Sources
- **NASA KOI**: Kepler Objects of Interest catalog (primary source)
- **K2 Mission**: Extended Kepler mission data (supplementary)  
- **TESS**: Transiting Exoplanet Survey Satellite (validation)

### 🎯 Target Classification
| Class | Description | Samples | Percentage |
|-------|-------------|---------|------------|
| **CONFIRMED** 🪐 | Verified exoplanets | 1,942 | 14.3% |
| **CANDIDATE** 🌍 | Potential exoplanets under investigation | 4,095 | 30.2% |
| **FALSE POSITIVE** ❌ | Objects incorrectly flagged as planets | 7,546 | 55.5% |
| **Total** | | **13,583** | **100%** |

### 🔧 Engineered Features (7 features)
| Feature | Description | Transformation | Importance |
|---------|-------------|----------------|------------|
| `period` | Orbital period (days) | Log-transformed | ⭐⭐⭐⭐⭐ |
| `radius` | Planet radius (Earth radii) | Normalized | ⭐⭐⭐⭐ |
| `temperature` | Stellar effective temperature (K) | StandardScaled | ⭐⭐⭐ |
| `insolation` | Insolation flux (Earth flux) | Log-transformed | ⭐⭐⭐⭐ |
| `depth` | Transit depth (ppm) | Log-transformed + clipping | ⭐⭐⭐⭐⭐ |
| `ra` | Right ascension (degrees) | Circular encoding | ⭐⭐ |
| `dec` | Declination (degrees) | Sine/cosine transformation | ⭐⭐ |

---

## 🏗️ Advanced Architecture

```
exoplanet-classifier/
├── 📁 data/
│   ├── raw/                    # Original NASA datasets (KOI, K2, TESS)
│   ├── processed/              # Engineered features and labels  
│   └── splits/                 # Stratified train/validation/test splits
├── 📁 src/
│   ├── data_processor.py       # Advanced data ingestion and cleaning
│   ├── feature_engineer.py     # Feature engineering and transformations
│   ├── train.py               # Basic training pipeline (legacy)
│   ├── enhanced_train.py      # 🆕 Advanced ML pipeline with Optuna
│   ├── predict.py             # Basic prediction interface (legacy)
│   └── enhanced_predict.py    # 🆕 Enterprise prediction pipeline
├── 📁 models/
│   ├── *_model.joblib         # Trained models (6 algorithms)
│   ├── *_scaler.joblib        # Feature scalers for each model
│   ├── label_encoder.joblib   # Label encoding mappings
│   └── metadata.json          # Model performance metrics & config
├── 📁 notebooks/
│   ├── eda.ipynb              # 🆕 Advanced EDA with PCA/t-SNE analysis
│   ├── model_evaluation.ipynb # 🆕 Comprehensive model comparison
│   └── experiments.ipynb      # Research and experimentation
├── 📁 reports/
│   ├── figures/               # Advanced visualizations (HTML/PNG)
│   └── predictions/           # Batch prediction results with confidence
├── app.py                     # Streamlit web application
├── requirements.txt           # Production dependencies
└── README.md                  # This comprehensive documentation
```

---

## 🛠️ Quick Start

### Prerequisites
- Python 3.8+
- 8GB+ RAM (recommended for advanced training)
- pip package manager

### Installation & Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/exoplanet-classifier.git
cd exoplanet-classifier

# Install dependencies (includes advanced ML packages)
pip install -r requirements.txt

# Download and process datasets
python src/data_processor.py

# Optional: Run advanced EDA (generates interactive visualizations)
jupyter notebook notebooks/eda.ipynb
```

### 🚀 Advanced Training Pipeline
```bash
# Run the enhanced training with all 6 algorithms + Optuna optimization
python src/enhanced_train.py

# This will:
# ✅ Train 6 different ML models
# ✅ Optimize hyperparameters with Optuna (100+ trials per model) 
# ✅ Create ensemble voting classifier
# ✅ Generate comprehensive performance reports
# ✅ Save all models and metadata
```

### 📊 Model Evaluation & Analysis
```bash
# Launch comprehensive model evaluation notebook
jupyter notebook notebooks/model_evaluation.ipynb

# Or run batch predictions with uncertainty estimation
python src/enhanced_predict.py
```

### 🌐 Web Application
```bash
# Launch the interactive Streamlit app
streamlit run app.py
# Navigate to: http://localhost:8501
```

---

## 🤖 Advanced ML Models

Our system implements 6 state-of-the-art machine learning algorithms:

| Algorithm | Type | Hyperparameters Tuned | Key Strengths |
|-----------|------|----------------------|---------------|
| **Random Forest** 🌲 | Ensemble | n_estimators, max_depth, min_samples_split | Robust, interpretable |
| **Gradient Boosting** 📈 | Boosting | n_estimators, learning_rate, max_depth | High performance |
| **XGBoost** ⚡ | Gradient Boosting | learning_rate, max_depth, subsample | Fast, accurate |
| **LightGBM** 💨 | Gradient Boosting | num_leaves, learning_rate, feature_fraction | Memory efficient |
| **SVM** 🎯 | Kernel Method | C, gamma, kernel type | Strong generalization |
| **Neural Network** 🧠 | Deep Learning | hidden_layers, learning_rate, dropout | Complex patterns |

### 🗳️ Ensemble Methods
- **Soft Voting**: Averages predicted probabilities for final decision
- **Hard Voting**: Uses majority vote from individual model predictions
- **Weighted Voting**: Applies performance-based weights to model contributions

---

## 📈 Performance Metrics & Evaluation

### 🎯 Model Comparison (Latest Results)
```
                   Accuracy  Precision  Recall   F1-Score
Ensemble (Soft)      0.724     0.698    0.715    0.706
XGBoost             0.719     0.692    0.708    0.700
LightGBM            0.716     0.689    0.705    0.697
Random Forest       0.695     0.670    0.685    0.677
Gradient Boosting   0.693     0.668    0.682    0.675
Neural Network      0.687     0.661    0.676    0.668
SVM                 0.681     0.655    0.670    0.662
```

### 📊 Advanced Evaluation Features
- **Cross-Validation**: 5-fold stratified validation for robust metrics
- **Learning Curves**: Training vs validation performance analysis  
- **Confusion Matrices**: Detailed classification breakdown by class
- **ROC/AUC Analysis**: Multi-class receiver operating characteristic
- **Feature Importance**: Model-specific and ensemble importance ranking
- **Uncertainty Estimation**: Prediction confidence and reliability scoring

---

## 🔍 Key Features & Capabilities

### 🧪 Advanced Data Analysis
- **Interactive Visualizations**: Plotly-powered dashboards and charts
- **Dimensionality Reduction**: PCA and t-SNE for high-dimensional data exploration  
- **Statistical Analysis**: Distribution analysis, outlier detection, correlation mapping
- **Feature Engineering**: Automated feature transformation and selection

### 🤖 Machine Learning Excellence
- **Automated Hyperparameter Tuning**: Optuna-based Bayesian optimization
- **Ensemble Learning**: Multiple model combination for improved accuracy
- **Cross-Validation**: Robust evaluation with stratified sampling
- **Model Interpretability**: Feature importance and prediction explanation

### 🚀 Production Ready
- **Scalable Architecture**: Modular design supporting easy extension
- **Comprehensive Logging**: Detailed execution tracking and debugging
- **Error Handling**: Robust exception handling and graceful degradation
- **Documentation**: Extensive docstrings and user guides

---

## 📚 Usage Examples

### Basic Prediction
```python
from src.enhanced_predict import EnhancedExoplanetPredictor

# Initialize predictor
predictor = EnhancedExoplanetPredictor()
predictor.load_models()

# Make predictions with uncertainty
results = predictor.predict_with_uncertainty(data)
print(f"Prediction: {results['prediction_labels'][0]}")
print(f"Confidence: {results['confidence'][0]:.3f}")
```

### Batch Processing
```python
# Process multiple samples with ensemble
results_df = predictor.batch_predict(
    data, 
    method='ensemble',
    save_results=True
)

# Generate comprehensive report
predictor.create_prediction_report(results_df)
```

### Custom Training
```python
from src.enhanced_train import EnhancedExoplanetTrainer

# Initialize trainer with custom configuration
trainer = EnhancedExoplanetTrainer(
    n_trials=150,  # More optimization trials
    cv_folds=10    # More thorough validation
)

# Train all models with optimization
trainer.train_all_models()
```

---

## 🎨 Visualizations & Reports

The system generates comprehensive visualizations:

### 📊 Interactive Dashboards
- **Model Comparison Dashboard**: Performance metrics across all algorithms
- **Feature Importance Analysis**: Interactive feature contribution charts
- **Prediction Confidence Maps**: Uncertainty visualization and reliability scoring
- **Learning Curve Analysis**: Training performance and overfitting detection

### 📈 Advanced Analytics
- **PCA Visualization**: Principal component analysis in 2D/3D space
- **t-SNE Clustering**: Non-linear dimensionality reduction for pattern discovery
- **Correlation Heatmaps**: Feature relationship analysis with statistical significance
- **Distribution Analysis**: Class balance and feature distribution insights

---

## 🤝 Contributing

We welcome contributions to enhance the exoplanet classification system!

### Development Areas
- **New Algorithms**: Implement additional ML models or deep learning architectures
- **Feature Engineering**: Develop new feature extraction and transformation methods  
- **Optimization**: Improve training speed and memory efficiency
- **Visualization**: Create new interactive charts and analysis tools
- **Documentation**: Enhance guides, tutorials, and code documentation

### Getting Started
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with comprehensive tests
4. Commit changes (`git commit -m 'Add amazing feature'`)
5. Push to branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request with detailed description

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **NASA Exoplanet Archive** for providing comprehensive datasets
- **Kepler, K2, and TESS missions** for groundbreaking exoplanet discoveries
- **NASA Space Apps Challenge** for inspiring innovative solutions
- **Open Source Community** for amazing tools and libraries

---

## 🔗 Links & Resources

- [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/)
- [Kepler Mission](https://www.nasa.gov/mission_pages/kepler/main/index.html)
- [TESS Mission](https://tess.gsfc.nasa.gov/)
- [NASA Space Apps Challenge](https://www.spaceappschallenge.org/)

---

**🌟 Star this repository if you find it useful for exoplanet research and machine learning!**

*Built with ❤️ for the NASA Space Apps Challenge 2025*