# Advanced Exoplanet Classification System
**NASA Space Apps Challenge 2025**

> An enterprise-grade machine learning solution for classifying astronomical objects as confirmed exoplanets, planet candidates, or false positives using data from NASA's Kepler, K2, and TESS missions. **Now optimized for RTX 4060 GPU acceleration!**

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![NASA](https://img.shields.io/badge/NASA-Space%20Apps%202025-red.svg)
![ML](https://img.shields.io/badge/ML-Advanced%20Ensemble-orange.svg)
![GPU](https://img.shields.io/badge/GPU-RTX%204060%20Ready-green.svg)

---

## 🚀 GPU Acceleration Setup (RTX 4060)

This project is optimized to leverage your RTX 4060 GPU for significantly faster training and inference. Follow these steps to enable GPU acceleration:

### Prerequisites
1. **NVIDIA Drivers**: Install latest GeForce drivers (536.xx or later)
2. **CUDA Toolkit**: Install CUDA 12.1 from [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)
3. **cuDNN**: Download and install cuDNN 8.9 for CUDA 12.1

### GPU Setup Commands
```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install TensorFlow with GPU support  
pip install tensorflow[and-cuda]

# Verify GPU detection
python check_gpu.py
```

### Quick GPU Check
Run this command to verify your RTX 4060 is detected:
```bash
python check_gpu.py
```

Expected output:
```
✅ PyTorch CUDA Available: True
✅ PyTorch GPU: NVIDIA GeForce RTX 4060
✅ TensorFlow GPUs Found: 1
🚀 Your RTX 4060 is ready for accelerated ML training!
```

---

## Project Overview

NASA's space missions have discovered thousands of potential exoplanets, but manual classification is time-intensive and prone to errors. Our AI system automates this process using advanced machine learning techniques to analyze stellar and planetary parameters.

### Key Features

- **Multi-Algorithm Ensemble**: Six advanced ML models with voting classifier
- **Hyperparameter Optimization**: Automated tuning with Optuna (100+ trials per model)
- **Uncertainty Estimation**: Confidence scoring and prediction reliability metrics
- **Production-Ready Architecture**: Modular, scalable, and fully documented codebase
- **Interactive Web Interface**: Real-time predictions with comprehensive visualizations
- **Model Explainability**: SHAP and LIME integration for interpretable AI

### Performance Highlights

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Ensemble (Best)** | **72.4%** | **71.8%** | **71.1%** | **71.4%** | **78.6%** |
| XGBoost | 71.9% | 71.2% | 70.5% | 70.8% | 78.2% |
| LightGBM | 71.6% | 70.8% | 70.2% | 70.5% | 77.9% |
| Random Forest | 68.2% | 67.5% | 66.8% | 67.1% | 74.5% |

---

## Dataset Information

### Data Sources
- **Kepler Objects of Interest (KOI)**: Primary exoplanet catalog
- **K2 Mission**: Extended Kepler mission data
- **TESS**: Transiting Exoplanet Survey Satellite observations

### Classification Categories
| Class | Description | Samples | Percentage |
|-------|-------------|---------|------------|
| **CONFIRMED** | Verified exoplanets | 1,942 | 14.3% |
| **CANDIDATE** | Potential exoplanets under investigation | 4,095 | 30.2% |
| **FALSE POSITIVE** | Objects incorrectly flagged as planets | 7,546 | 55.5% |

**Total Dataset**: 13,583 samples with comprehensive feature engineering

### Engineered Features
- **period**: Orbital period (days) - Log-transformed
- **radius**: Planet radius (Earth radii) - Normalized
- **temperature**: Stellar effective temperature (K) - StandardScaled
- **insolation**: Insolation flux (Earth flux) - Log-transformed
- **depth**: Transit depth (ppm) - Log-transformed and clipped
- **ra**: Right ascension (degrees) - Circular encoding
- **dec**: Declination (degrees) - Sine/cosine transformation

---

## Architecture

```
exoplanet-classifier/
├── data/
│   ├── raw/                    # Original NASA datasets
│   ├── processed/              # Engineered features and labels
│   └── splits/                 # Stratified train/validation/test splits
├── src/
│   ├── data_processor.py       # Data ingestion and preprocessing
│   ├── feature_engineer.py     # Advanced feature engineering
│   ├── enhanced_train.py       # Advanced ML training pipeline
│   ├── enhanced_predict.py     # Enterprise prediction system
│   └── explainability.py      # Model interpretation tools
├── models/
│   ├── *.pkl                   # Trained model artifacts
│   ├── *.joblib               # Model persistence files
│   └── *.json                 # Model metadata and configurations
├── notebooks/
│   ├── eda.ipynb              # Advanced exploratory data analysis
│   └── model_evaluation.ipynb # Comprehensive model evaluation
├── reports/                    # Generated analysis reports
├── app.py                     # Streamlit web application
└── requirements.txt           # Production dependencies
```

---

## Quick Start

### Prerequisites
- Python 3.8 or higher
- 8GB+ RAM (recommended for training)
- pip package manager

### Installation
```bash
# Clone repository
git clone https://github.com/Harshad2321/Exoplanet-Classifier-NASA-KOI-K2-TESS-.git
cd Exoplanet-Classifier-NASA-KOI-K2-TESS-

# Install dependencies
pip install -r requirements.txt

# Launch web application
streamlit run app.py
```

### Advanced Training
```bash
# Train all models with optimization
python src/enhanced_train.py

# Evaluate model performance
jupyter notebook notebooks/model_evaluation.ipynb

# Run predictions with uncertainty estimation
python src/enhanced_predict.py
```

---

## Machine Learning Pipeline

### Algorithms Implemented
1. **Random Forest**: Ensemble method with feature importance
2. **Gradient Boosting**: Sequential error correction learning
3. **XGBoost**: Extreme gradient boosting with regularization
4. **LightGBM**: Memory-efficient gradient boosting
5. **Support Vector Machine**: Kernel-based classification
6. **Neural Network**: Multi-layer perceptron with dropout

### Advanced Features
- **Hyperparameter Tuning**: Bayesian optimization with Optuna
- **Ensemble Methods**: Soft/hard voting for improved accuracy
- **Cross-Validation**: Stratified 5-fold validation
- **Uncertainty Quantification**: Entropy-based confidence scoring
- **Feature Engineering**: Advanced transformations and encoding
- **Model Persistence**: Automated saving and loading

---

## Web Application

The Streamlit web interface provides:

- **Real-time Predictions**: Interactive parameter input with instant results
- **Model Comparison**: Performance metrics across all algorithms
- **Uncertainty Analysis**: Confidence scoring and reliability metrics
- **Explainability Tools**: Feature importance and prediction explanations
- **Data Visualization**: Interactive charts and analysis dashboards

### Usage Example
```python
# Import prediction system
from src.enhanced_predict import EnhancedExoplanetPredictor

# Initialize predictor
predictor = EnhancedExoplanetPredictor()
predictor.load_models()

# Make prediction with uncertainty
results = predictor.predict_with_uncertainty(input_data)
print(f"Prediction: {results['prediction_labels'][0]}")
print(f"Confidence: {results['confidence'][0]:.2%}")
```

---

## Model Evaluation

### Comprehensive Metrics
- **Classification Metrics**: Accuracy, precision, recall, F1-score
- **Probabilistic Metrics**: ROC-AUC, precision-recall curves
- **Ensemble Analysis**: Model agreement and voting patterns
- **Learning Curves**: Training vs validation performance
- **Feature Importance**: Global and local explanations

### Evaluation Framework
The system includes automated evaluation with:
- Confusion matrices for detailed error analysis
- ROC curves for threshold optimization
- Feature importance rankings across models
- Cross-validation stability metrics
- Uncertainty calibration analysis

---

## Deployment

### Streamlit Cloud (Recommended)
1. Push code to GitHub repository
2. Connect to [share.streamlit.io](https://share.streamlit.io)
3. Deploy with main file: `app.py`
4. Access live demo at generated URL

### Docker Deployment
```bash
# Build container
docker build -t exoplanet-classifier .

# Run application
docker run -p 8501:8501 exoplanet-classifier
```

### Local Development
```bash
# Install in development mode
pip install -e .

# Run with hot reload
streamlit run app.py --server.runOnSave true
```

---

## Contributing

We welcome contributions to enhance the exoplanet classification system:

### Development Areas
- **Algorithm Implementation**: New ML models or ensemble methods
- **Feature Engineering**: Advanced transformations and selections
- **Visualization**: Interactive charts and analysis tools
- **Performance**: Optimization and scalability improvements
- **Documentation**: Technical guides and tutorials

### Getting Started
1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push branch: `git push origin feature-name`
5. Submit pull request with detailed description

---

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- **NASA Exoplanet Archive** for comprehensive datasets
- **Kepler, K2, and TESS missions** for groundbreaking discoveries
- **NASA Space Apps Challenge** for inspiring innovation
- **Open source community** for exceptional tools and libraries

---

## Links and Resources

- [Live Demo](https://your-app.streamlit.app) (Deploy to get link)
- [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/)
- [NASA Space Apps Challenge](https://www.spaceappschallenge.org/)
- [Documentation](docs/) (Coming soon)

---

**Built for NASA Space Apps Challenge 2025 - Advancing exoplanet discovery through artificial intelligence**