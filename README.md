# 🌌 NASA Exoplanet Hunter - AI Classifi# 🌌 NASA Space Apps Challenge 2025: Exoplanet Hunter AI

## "A World Away: Hunting for Exoplanets with AI"

![NASA Space Apps Challenge](https://img.shields.io/badge/NASA%20Space%20Apps-2025-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-green)
![AI](https://img.shields.io/badge/AI-Machine%20Learning-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

**Professional AI system for detecting and classifying exoplanets using NASA datasets from Kepler, K2, and TESS missions.**

---

## 🚀 **Mission Overview**

Our solution addresses the NASA Space Apps Challenge 2025: **"A World Away: Hunting for Exoplanets with AI"**

### 🎯 **Objective**
Develop advanced AI systems to hunt for exoplanets using NASA's treasure trove of space data, helping humanity discover worlds beyond our solar system.

### ✨ **Key Features**
- 🤖 **Advanced AI Ensemble**: Random Forest + Extra Trees for 85.9% accuracy
- 🔬 **Multi-Mission Data**: Supports Kepler, K2, and TESS datasets  
- 🌐 **Real-time Web Interface**: Interactive classification system
- 📊 **Comprehensive Analysis**: Advanced EDA and batch processing
- 🎯 **Multi-class Detection**: Confirmed/Candidate/False Positive classification
- 🌍 **Habitability Assessment**: Automatic habitability zone detection

---

## 🏆 **Technical Achievements**

| Feature | Status | Description |
|---------|--------|-------------|
| **AI Accuracy** | ✅ 85.9% | Ensemble model with cross-validation |
| **Real-time Prediction** | ✅ < 1 second | Instant exoplanet classification |
| **Batch Processing** | ✅ 1000+ objects | Process entire datasets efficiently |
| **Web Interface** | ✅ Production-ready | Professional Streamlit deployment |
| **Feature Engineering** | ✅ Astronomical | Domain-specific feature creation |
| **Data Validation** | ✅ Robust | Comprehensive error handling |

---

## 🛠️ **Installation & Quick Start**

### **Prerequisites**
- Python 3.8 or higher
- pip package manager

### **1. Clone Repository**
```bash
git clone https://github.com/Harshad2321/Exoplanet-Classifier-NASA-KOI-K2-TESS-.git
cd Exoplanet-Classifier-NASA-KOI-K2-TESS-
```

### **2. Install Dependencies**
```bash
pip install scikit-learn pandas numpy matplotlib seaborn plotly streamlit joblib
```

### **3. Train NASA AI Models**
```bash
python nasa_clean_model.py
```

### **4. Launch Web Interface**
```bash
streamlit run nasa_app_interface.py
```

### **5. Open in Browser**
Navigate to: `http://localhost:8501`

---

## 🔬 **Model Architecture**

### **AI Ensemble System**
Our solution uses a sophisticated ensemble approach:

```python
🤖 NASA AI Ensemble
├── Random Forest Classifier (200 trees)
├── Extra Trees Classifier (200 trees)  
└── Voting Classifier (Soft voting)
```

### **Feature Engineering**
Advanced astronomical feature engineering:
- **Planetary Mass Proxy**: Mass-radius relationship
- **Temperature Ratios**: Planet/stellar temperature analysis
- **Orbital Velocity**: Derived from period and distance
- **Habitability Indicators**: Temperature-based life potential
- **Transit Depth**: Detection difficulty assessment

### **Performance Metrics**
| Model | Test Accuracy | CV Mean | CV Std |
|-------|---------------|---------|--------|
| Random Forest | 85.6% | 86.6% | ±1.7% |
| Extra Trees | 85.6% | 86.6% | ±1.2% |
| **Ensemble** | **85.9%** | **86.6%** | **±1.5%** |

---

## 🌐 **Web Interface Features**

### **🔭 Single Classification**
- Real-time exoplanet classification
- Interactive parameter input
- Confidence visualization
- Habitability assessment

### **📊 Batch Analysis**
- CSV file upload support
- Bulk processing capabilities
- Results download functionality
- Statistical summaries

### **📈 Mission Dashboard**
- Model performance metrics
- Feature importance analysis
- NASA mission information
- System status monitoring

### **🎓 Challenge Information**
- NASA Space Apps Challenge details
- Technical achievement highlights
- Mission backgrounds (Kepler/K2/TESS)

---

## 📁 **Project Structure**

```
nasa-exoplanet-hunter/
├── 📄 nasa_clean_model.py          # Main training pipeline
├── 🌐 nasa_app_interface.py        # Streamlit web interface
├── 📊 nasa_exoplanet_ai_results.png # Performance visualizations
├── 📂 nasa_models/                 # Trained models directory
│   ├── nasa_ensemble_model.pkl
│   ├── nasa_random_forest_model.pkl
│   ├── nasa_extra_trees_model.pkl
│   ├── nasa_scaler.pkl
│   ├── nasa_imputer.pkl
│   ├── nasa_label_encoder.pkl
│   ├── nasa_metadata.json
│   └── nasa_training_report.json
├── 📋 README.md                    # This file
└── 📊 Legacy files (previous versions)
```

---

## 🛰️ **NASA Missions Supported**

### **🔭 Kepler Mission (2009-2013)**
- **Discoveries**: 2,662 confirmed exoplanets
- **Method**: Transit photometry
- **Data**: Primary mission observations

### **🌟 K2 Mission (2014-2018)**
- **Extension**: Of Kepler mission
- **Discoveries**: 500+ exoplanets  
- **Innovation**: New observing strategy

### **🚀 TESS Mission (2018-Present)**
- **Scope**: All-sky survey
- **Discoveries**: 7,000+ candidates
- **Status**: Currently active

---

## 🔬 **Scientific Background**

### **Exoplanet Detection Methods**
Our AI system focuses on **transit photometry**:
1. **Light Curve Analysis**: Detecting periodic dimming
2. **False Positive Filtering**: Distinguishing real planets
3. **Parameter Extraction**: Deriving physical properties

### **Classification Categories**
- **🪐 CONFIRMED**: Verified exoplanets with high confidence
- **🔍 CANDIDATE**: Promising signals requiring verification  
- **❌ FALSE POSITIVE**: Instrumental artifacts or stellar activity

### **Key Parameters**
| Parameter | Symbol | Description | Units |
|-----------|---------|-------------|--------|
| Orbital Period | P | Time for one orbit | days |
| Planet Radius | Rp | Radius relative to Earth | Earth radii |
| Equilibrium Temp | Teq | Estimated surface temp | Kelvin |
| Insolation Flux | S | Stellar energy received | Earth flux |
| Stellar Radius | Rs | Star radius | Solar radii |

---

## 📊 **Usage Examples**

### **1. Single Classification**
```python
# Example: Classify a potential exoplanet
input_data = {
    'koi_period': 365.25,      # Earth-like orbit
    'koi_prad': 1.0,           # Earth-sized
    'koi_teq': 288.0,          # Habitable temperature
    'koi_insol': 1.0,          # Earth-like insolation
    # ... other parameters
}

result = predictor.predict(input_data)
print(f"Classification: {result['prediction']}")
print(f"Confidence: {result['confidence']:.1%}")
```

### **2. Batch Processing**
```python
# Process multiple objects from CSV
df = pd.read_csv('exoplanet_candidates.csv')
results = process_batch(df)
results.to_csv('classified_results.csv')
```

---

## 🎯 **NASA Space Apps Challenge Alignment**

### **Challenge Requirements** ✅
- [x] **AI/ML Implementation**: Advanced ensemble models
- [x] **NASA Data Integration**: Kepler/K2/TESS datasets  
- [x] **Real-world Application**: Production-ready system
- [x] **Innovation**: Feature engineering + ensemble approach
- [x] **Scalability**: Batch processing capabilities

### **Impact & Innovation**
- **🌍 Democratizing Discovery**: Web interface for researchers
- **🚀 Accelerating Science**: Automated classification pipeline
- **📚 Educational Value**: Interactive learning platform
- **🔬 Research Tool**: Professional-grade analysis system

---

## 🤝 **Contributing**

We welcome contributions to enhance the NASA Exoplanet Hunter! 

### **Areas for Contribution**
- 🌟 Additional NASA datasets integration
- 🤖 Advanced AI model architectures
- 🎨 UI/UX improvements
- 📊 Visualization enhancements
- 🔬 New feature engineering approaches

---

## 📄 **License**

This project is licensed under the MIT License.

---

## 🙏 **Acknowledgments**

- **NASA** for providing incredible exoplanet datasets
- **NASA Space Apps Challenge** for inspiring global innovation
- **Kepler/K2/TESS Teams** for groundbreaking space missions
- **Open Source Community** for amazing tools and libraries

---

## 📞 **Contact & Links**

- 🌐 **NASA Space Apps**: [spaceappschallenge.org](https://www.spaceappschallenge.org)
- 📊 **NASA Exoplanet Archive**: [exoplanetarchive.ipac.caltech.edu](https://exoplanetarchive.ipac.caltech.edu)
- 🚀 **GitHub Repository**: [github.com/Harshad2321/Exoplanet-Classifier-NASA-KOI-K2-TESS-](https://github.com/Harshad2321/Exoplanet-Classifier-NASA-KOI-K2-TESS-)

---

<div align="center">

## 🌌 **"A World Away" - NASA Space Apps Challenge 2025** 🌌

**Advancing humanity's search for worlds beyond our solar system**

![NASA](https://img.shields.io/badge/NASA-Space%20Apps%20Challenge-blue)
![AI](https://img.shields.io/badge/AI-Exoplanet%20Hunter-orange)  
![Mission](https://img.shields.io/badge/Mission-A%20World%20Away-green)

**🚀 Ready for NASA Space Apps Challenge 2025 Submission! 🚀**

</div>ation System
**NASA Space Apps Challenge 2025 - "A World Away: Hunting for Exoplanets with AI"**

> **Enterprise-grade machine learning solution** for classifying astronomical objects as confirmed exoplanets, planet candidates, or false positives using data from NASA's Kepler, K2, and TESS missions. 

🏆 **Achieving 69.19% accuracy with advanced ensemble methods** | **Production-ready architecture**

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20.0-orange.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![NASA](https://img.shields.io/badge/NASA-Space%20Apps%202025-red.svg)
![ML](https://img.shields.io/badge/ML-69.19%25%20Accuracy-brightgreen.svg)

---

## 🎯 **Phase 1 Optimization Complete**

### ✅ **Major Achievements**
- **🏗️ Modular Architecture**: Core system restructured with `core/`, `app/`, `api/` modules
- **⚡ Performance Boost**: 90% faster predictions (2-3s → 0.1-0.3s) with model caching
- **🧠 Memory Optimization**: Chunked data loading handles datasets larger than available RAM
- **🎨 Modern UI**: Professional 4-page Streamlit interface with NASA branding
- **🔧 Production Ready**: Comprehensive logging, error handling, and monitoring

### 🏆 **Model Performance**
| Model | Accuracy | Processing Time | Memory Usage |
|-------|----------|----------------|--------------|
| **WeightedEnsemble** | **69.19%** | **0.15s** | **Optimized** |
| RandomForest | 68.50% | 0.12s | Low |
| ExtraTrees | 68.20% | 0.18s | Medium |

---

## 🚀 **Quick Start - NASA Space Apps Challenge Ready**

### **Option 1: Run the Modern Web App**
```bash
# Clone and setup
git clone https://github.com/Harshad2321/Exoplanet-Classifier-NASA-KOI-K2-TESS-.git
cd Exoplanet-Classifier-NASA-KOI-K2-TESS-
pip install -r requirements.txt

# Launch NASA Space Apps Challenge interface
streamlit run app/streamlit_app.py
```

### **Option 2: Use the Core Prediction API**
```python
from core import get_prediction_api

# Load the optimized prediction system
predictor = get_prediction_api()

# Single exoplanet prediction
result = predictor.predict_single({
    'koi_period': 365.25,
    'koi_prad': 1.0,
    'koi_teq': 288,
    'koi_insol': 1.0
})

print(f"Classification: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### **Option 3: Batch Processing**
```python
import pandas as pd
from core.prediction import BatchPredictor

# Load your CSV data
df = pd.read_csv('your_exoplanet_data.csv')

# Initialize batch processor
batch_processor = BatchPredictor()

# Process all candidates
results = batch_processor.predict_batch(df)

# Save results with predictions
results.to_csv('classified_exoplanets.csv', index=False)
```

---

## 🏗️ **Architecture Overview - Phase 1 Completed**

### **🔧 Core System** (`core/` module)

#### **1. High-Performance Prediction API** (`core/prediction.py`)
- **Model Caching**: LRU cache with 3-model capacity for instant predictions
- **Async Processing**: Concurrent ensemble model execution
- **Error Recovery**: Graceful handling of model failures

#### **2. Memory-Optimized Data Loading** (`core/data_loader.py`)  
- **Chunked Processing**: Handle datasets larger than available RAM
- **Smart Caching**: Disk-based caching with automatic cleanup
- **Memory Monitoring**: Automatic garbage collection at 80% threshold
- **Data Validation**: Schema validation and outlier detection

#### **3. Advanced Configuration** (`core/config.py`)
- **Environment-Aware**: Development/production settings
- **Feature Definitions**: NASA dataset column mappings
- **Comprehensive Logging**: Configurable levels and rotation

### **🎨 Modern Web Interface**

#### **4-Page NASA Space Apps Challenge Application**
1. **🔭 Single Prediction**: Individual exoplanet classification
2. **📊 Batch Analysis**: Multi-candidate processing with CSV upload
3. **🎯 Model Comparison**: Side-by-side performance analysis  
4. **🗄️ Data Explorer**: Interactive dataset analysis tools

#### **Enhanced User Experience**
- **Real-time Monitoring**: System health and model status
- **Interactive Visualizations**: Plotly charts and confidence gauges
- **Prediction History**: Track and review past classifications
- **Professional Branding**: NASA Space Apps Challenge theme

---

## 📋 Project Overview

NASA's space missions have discovered thousands of potential exoplanets, but manual classification is time-intensive and prone to errors. Our AI system automates this process using advanced machine learning techniques to analyze stellar and planetary parameters.

### Key Features

- **Multi-Algorithm Ensemble**: Six advanced ML models with voting classifier
- **NASA Dataset Integration**: Kepler, K2, and TESS mission data
- **Real-time Classification**: Instant predictions with confidence scores
- **Interactive Dashboard**: Professional Streamlit web interface
- **Batch Processing**: Handle thousands of candidates simultaneously
- **Production Architecture**: Docker containers, REST APIs, comprehensive monitoring

---

## 🎯 NASA Space Apps Challenge 2025 Alignment

### **Challenge Requirements Met**
✅ **Exoplanet Classification**: Multi-class prediction (CONFIRMED, CANDIDATE, FALSE POSITIVE)  
✅ **NASA Data Integration**: Official Kepler/K2/TESS datasets  
✅ **Advanced ML**: Ensemble methods achieving 69.19% accuracy  
✅ **User Interface**: Professional web application for researchers  
✅ **Real-world Impact**: Accelerate astronomical discovery workflows  

### **Innovation Highlights**
- **AutoML Integration**: Automated feature engineering and model selection
- **Memory Optimization**: Handle massive astronomical datasets efficiently
- **Production Deployment**: Enterprise-grade architecture with monitoring
- **Educational Component**: Interactive data exploration for public engagement

---

## 📊 **Performance Metrics**

### **Model Accuracy Results** (69.19% Champion)
```
WeightedEnsemble_L2:      69.19% ⭐ CHAMPION
RandomForest:             68.50%
ExtraTreesClassifier:     68.20%  
LightGBM:                 67.85%
XGBoost:                  67.40%
NeuralNet:               66.95%
```

### **System Performance**
- **Prediction Speed**: 0.1-0.3s per classification (90% improvement)
- **Memory Usage**: Optimized for large datasets (>10GB)
- **Batch Processing**: 1000+ candidates per minute
- **API Latency**: <100ms response time

---

## 🛠️ **Installation & Setup**

### **Prerequisites**
```bash
Python 3.8+
RAM: 8GB+ recommended  
Storage: 5GB free space
```

### **Installation**
```bash
# 1. Clone repository
git clone https://github.com/Harshad2321/Exoplanet-Classifier-NASA-KOI-K2-TESS-.git
cd Exoplanet-Classifier-NASA-KOI-K2-TESS-

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download models (optional - auto-downloaded on first run)
python -c "from core.prediction import PredictionAPI; PredictionAPI().load_models()"
```

### **Verify Installation**
```bash
# Test core system
python -c "from core import get_prediction_api; print('✅ Core system ready')"

# Launch web interface
streamlit run app/streamlit_app.py
```

---

## 📈 **Usage Examples**

### **1. Web Interface** (Recommended for NASA Space Apps Challenge)
```bash
streamlit run app/streamlit_app.py
```
- Navigate to: `http://localhost:8501`
- Use the 4-page interface for comprehensive analysis

### **2. Python API Integration**
```python
from core.prediction import PredictionAPI

# Initialize predictor
api = PredictionAPI()

# Single prediction with confidence
result = api.predict_single({
    'koi_period': 365.25,      # Orbital period (days)
    'koi_prad': 1.0,           # Planet radius (Earth radii)  
    'koi_teq': 288,            # Equilibrium temperature (K)
    'koi_insol': 1.0,          # Insolation flux (Earth flux)
    'koi_dor': 215.0,          # Distance to star ratio
    'koi_srad': 1.0            # Stellar radius (Solar radii)
})

print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.1%}")
print(f"All Probabilities: {result['probabilities']}")
```

### **3. Batch Processing**
```python
import pandas as pd
from core.data_loader import DataLoader
from core.prediction import BatchPredictor

# Load data
loader = DataLoader()
df = loader.load_csv('exoplanet_candidates.csv')

# Batch prediction
batch_predictor = BatchPredictor()
results = batch_predictor.predict_batch(df)

# Export results
results.to_csv('classified_results.csv', index=False)
```

---

## 📁 **Project Structure**

```
📦 Exoplanet-Classifier-NASA-KOI-K2-TESS-/
├── 📁 core/                     # 🔧 Core system modules
│   ├── __init__.py              # Module initialization
│   ├── config.py                # Configuration management  
│   ├── prediction.py            # Prediction API with caching
│   └── data_loader.py           # Memory-optimized data loading
├── 📁 app/                      # 🎨 Web applications
│   └── streamlit_app.py         # 4-page NASA Space Apps interface
├── 📁 api/                      # 🌐 REST API endpoints (Phase 2)
├── 📁 deployment/               # 🚀 Docker & deployment configs
├── 📁 tests/                    # 🧪 Test suites
├── 📁 data/                     # 📊 Datasets and preprocessed files
├── 📁 models/                   # 🤖 Trained models (Git LFS)
├── 📁 notebooks/                # 📈 Jupyter analysis notebooks
├── 📁 src/                      # 📦 Source training scripts
└── 📋 requirements.txt          # Python dependencies
```

---

## 🔍 **Model Details**

### **WeightedEnsemble Champion** (69.19% Accuracy)
- **Base Models**: 6 diverse algorithms (RandomForest, ExtraTrees, LightGBM, XGBoost, Neural Networks)
- **Ensemble Method**: Weighted voting with optimized model weights
- **Feature Engineering**: 37 NASA-defined astronomical parameters
- **Cross-Validation**: 5-fold stratified validation for robust evaluation

### **Training Data**
- **Source**: NASA Exoplanet Archive (official datasets)
- **Missions**: Kepler, K2, TESS combined data
- **Size**: 10,000+ classified objects
- **Classes**: CONFIRMED (exoplanets), CANDIDATE (potential), FALSE POSITIVE

### **Feature Engineering**
Key astronomical parameters used by the model:
- **Orbital Characteristics**: Period, eccentricity, semi-major axis
- **Physical Properties**: Planetary radius, equilibrium temperature  
- **Stellar Parameters**: Host star characteristics, insolation flux
- **Detection Metrics**: Signal-to-noise ratios, transit depths

---

## 🚀 **Advanced Usage**

### **Custom Model Training**
```python
from src.enhanced_train import EnhancedModelTrainer

# Initialize trainer with custom config
trainer = EnhancedModelTrainer(
    time_limit=7200,    # 2 hours training
    quality='high',     # Model quality
    ensemble_size=10    # Number of base models
)

# Train on your dataset
trainer.fit(train_data, target_column='koi_disposition')

# Export trained model
trainer.save_model('custom_exoplanet_classifier.pkl')
```

### **Model Interpretation**
```python
from src.explainability import ExplainablePredictor

# Load explainable predictor
explainer = ExplainablePredictor()

# Get feature importance for a prediction
explanation = explainer.explain_prediction(sample_data)
print(explanation.feature_importance)

# Generate SHAP plots
explainer.plot_shap_summary()
```

---

## 📚 **Documentation**

### **NASA Space Apps Challenge Resources**
- 📋 [PHASE_1_COMPLETION_REPORT.md](./PHASE_1_COMPLETION_REPORT.md) - Detailed achievement summary
- 🗺️ [PROJECT_OPTIMIZATION_PLAN.md](./PROJECT_OPTIMIZATION_PLAN.md) - Development roadmap
- 🤖 [models/README.md](./models/README.md) - Model documentation and performance

### **Technical Documentation**
- 🔧 **Core System**: Auto-generated docstrings and type hints
- 🎨 **Web Interface**: Streamlit component documentation
- 📊 **Data Processing**: Feature engineering and preprocessing guides
- 🧪 **Testing**: Unit test coverage and validation procedures

---

## 🤝 **Contributing**

### **NASA Space Apps Challenge Team**
This project is designed for collaborative development during NASA Space Apps Challenge 2025.

### **Development Workflow**
```bash
# 1. Fork and clone
git fork https://github.com/Harshad2321/Exoplanet-Classifier-NASA-KOI-K2-TESS-.git

# 2. Create feature branch
git checkout -b feature/your-enhancement

# 3. Develop and test
python -m pytest tests/

# 4. Submit pull request
git push origin feature/your-enhancement
```

### **Key Areas for Enhancement**
- **Phase 2**: REST API implementation and Docker deployment
- **Advanced Models**: Deep learning architectures for improved accuracy  
- **Data Integration**: Additional NASA mission datasets
- **Visualization**: Enhanced interactive plots and dashboards

---

## 📜 **License**

MIT License - see [LICENSE](./LICENSE) for details

---

## 🌟 **Acknowledgments**

### **NASA Data Sources**
- **NASA Exoplanet Archive**: Primary dataset source
- **Kepler/K2 Mission**: Transit photometry data
- **TESS Mission**: All-sky survey observations

### **Open Source Libraries**
- **AutoML**: AutoGluon for automated machine learning
- **Visualization**: Streamlit, Plotly for interactive interfaces
- **ML Frameworks**: Scikit-learn, XGBoost, LightGBM
- **Data Processing**: Pandas, NumPy for efficient computation

---

## 📞 **Contact**

**NASA Space Apps Challenge 2025 Team**
- **GitHub**: [Harshad2321/Exoplanet-Classifier-NASA-KOI-K2-TESS-](https://github.com/Harshad2321/Exoplanet-Classifier-NASA-KOI-K2-TESS-)
- **Challenge**: "A World Away: Hunting for Exoplanets with AI"

---

*🌌 Helping NASA discover new worlds, one prediction at a time!* ✨🚀