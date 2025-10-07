
# NASA Exoplanet Classifier

![NASA Space Apps Challenge](https://img.shields.io/badge/NASA%20Space%20Apps-2025-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-green)
![AI](https://img.shields.io/badge/AI-Machine%20Learning-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

**Professional AI system for detecting and classifying exoplanets using NASA datasets from Kepler, K2, and TESS missions.**

## Overview

Our solution addresses the NASA Space Apps Challenge 2025: **"A World Away: Hunting for Exoplanets with AI"**

Develop advanced AI systems to hunt for exoplanets using NASA's treasure trove of space data, helping humanity discover worlds beyond our solar system.

### Key Features

- **Advanced AI Ensemble**: Random Forest + Extra Trees for 85.9% accuracy
- **Multi-Mission Data**: Supports Kepler, K2, and TESS datasets  
- **Full Stack Application**: React frontend + FastAPI backend
- **Real-time Classification**: Interactive web interface
- **Smart Model Selection**: Automatic AI model optimization
- **Batch Processing**: Handle multiple exoplanet classifications
- **Multi-class Detection**: Confirmed/Candidate/False Positive classification

### Performance Metrics

| Feature | Status | Description |
|---------|--------|-------------|
| **AI Accuracy** | 85.9% | Ensemble model with cross-validation |
| **Real-time Prediction** | < 500ms | Instant exoplanet classification |
| **Batch Processing** | 1000+ objects | Process entire datasets efficiently |
| **Web Interface** | Production-ready | Modern React + TypeScript frontend |
| **API Response** | RESTful | FastAPI with automatic documentation |
| **Data Validation** | Robust | Comprehensive error handling |

## Quick Start

### Prerequisites

- Python 3.8 or higher
- Node.js 14+ (for frontend development)
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Harshad2321/Exoplanet-Classifier-NASA-KOI-K2-TESS-.git
cd Exoplanet-Classifier-NASA-KOI-K2-TESS-
```

2. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

3. **Start the application**
```bash
python start_app.py
```

The application will start with:
- **Streamlit Interface**: http://localhost:8501
- **FastAPI Backend**: http://localhost:8000  
- **API Documentation**: http://localhost:8000/docs

### Alternative: Run Streamlit Only

For Streamlit interface only:
```bash
streamlit run nasa_app_interface.py
```

Navigate to: `http://localhost:8501`

### Docker Deployment

```bash
docker build -t nasa-exoplanet-classifier .
docker run -p 8000:8000 nasa-exoplanet-classifier
```

## Architecture

### Full Stack Application

Our solution uses a modern full-stack architecture:

**Frontend (React + TypeScript)**
- Modern UI with space-themed design
- Real-time classification interface
- Interactive parameter inputs
- Batch processing capabilities
- Results visualization

**Backend (FastAPI + Python)**  
- RESTful API endpoints
- Smart AI model selection
- Real-time predictions
- Batch processing support
- Automatic API documentation

### AI Model Ensemble

```python
NASA AI Ensemble Architecture:
├── Random Forest Classifier (200 trees)
├── Extra Trees Classifier (200 trees)  
└── Voting Classifier (Soft voting)
```

**Advanced Feature Engineering:**
- **Planetary Mass Proxy**: Mass-radius relationship analysis
- **Temperature Ratios**: Planet/stellar temperature comparison  
- **Orbital Velocity**: Derived from period and distance
- **Habitability Indicators**: Temperature-based life potential
- **Transit Depth**: Detection difficulty assessment

### Performance Metrics

| Model | Test Accuracy | CV Mean | CV Std |
|-------|---------------|---------|--------|
| Random Forest | 85.6% | 86.6% | ±1.7% |
| Extra Trees | 85.6% | 86.6% | ±1.2% |
| **Ensemble** | **85.9%** | **86.6%** | **±1.5%** |

## Usage

### Web Interface Features

**Single Classification:**
- Real-time exoplanet classification
- Interactive parameter input
- Confidence visualization  
- Habitability assessment

**Batch Processing:**
- CSV file upload support
- Bulk processing capabilities
- Results download functionality
- Statistical summaries

**Model Analytics:**
- Performance metrics dashboard
- Feature importance analysis  
- Smart AI model selection
- System status monitoring

### API Endpoints

**FastAPI Backend** (`http://localhost:8000`):
- `POST /classify` - Single exoplanet classification
- `POST /batch-classify` - Batch processing
- `GET /health` - System health check
- `GET /docs` - Interactive API documentation

## Project Structure

```
nasa-exoplanet-classifier/
├── backend_api.py              # FastAPI backend server
├── nasa_app_interface.py       # Streamlit web interface  
├── nasa_smart_classifier.py    # Smart AI model selection
├── nasa_clean_model.py         # Model training utilities
├── start_app.py                # Application launcher
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Container configuration
├── frontend/                   # React frontend application
│   ├── App.tsx                # Main React application
│   ├── components/            # React components
│   ├── package.json           # Node.js dependencies
│   └── vite.config.ts         # Build configuration
└── nasa_models/               # Trained AI models
    ├── nasa_ensemble_model.pkl
    ├── nasa_random_forest_model.pkl  
    ├── nasa_extra_trees_model.pkl
    ├── nasa_scaler.pkl
    ├── nasa_imputer.pkl
    ├── nasa_label_encoder.pkl
    └── nasa_metadata.json
```

## NASA Missions Data

**Kepler Mission:**
- **Discoveries**: 2,662 confirmed exoplanets
- **Method**: Transit photometry
- **Data**: Primary mission observations (2009-2013)

**K2 Mission:**  
- **Extension**: Of Kepler mission
- **Discoveries**: 500+ additional exoplanets
- **Innovation**: New observing strategy (2014-2018)

**TESS Mission:**
- **Scope**: All-sky survey
- **Discoveries**: 7,000+ candidates
- **Status**: Currently active (2018-present)
- **Discoveries**: 1000+ confirmed exoplanets and growing

## Detection Method

Our AI system focuses on **transit photometry analysis**:

1. **Light Curve Analysis**: Detecting periodic dimming patterns
2. **False Positive Filtering**: Distinguishing real planets from noise
3. **Parameter Extraction**: Deriving physical properties from signals
4. **Habitability Assessment**: Evaluating potential for life

### Classification Categories

- **CONFIRMED**: Verified exoplanets with high confidence (>95%)
- **CANDIDATE**: Promising signals requiring further verification  
- **FALSE POSITIVE**: Instrumental artifacts or stellar activity

### Key Parameters

| Parameter | Symbol | Description | Units |
|-----------|---------|-------------|--------|
| Orbital Period | P | Time for one complete orbit | days |
| Planet Radius | Rp | Radius relative to Earth | Earth radii |
| Equilibrium Temp | Teq | Estimated surface temperature | Kelvin |
| Insolation Flux | S | Stellar energy received | Earth flux |
| Stellar Radius | Rs | Host star radius | Solar radii |
| Impact Parameter | b | Transit geometry | dimensionless |

## Code Examples

### Single Classification

```python
from nasa_smart_classifier import SmartNASAExoplanetClassifier

# Initialize classifier
classifier = SmartNASAExoplanetClassifier()

# Input exoplanet parameters
input_data = {
    'koi_period': 365.25,      # Earth-like orbital period
    'koi_prad': 1.0,           # Earth-size radius
    'koi_teq': 288.0,          # Habitable zone temperature
    'koi_insol': 1.0,          # Earth-like stellar flux
    'koi_srad': 1.0,           # Sun-like star
    'koi_slogg': 4.5           # Solar surface gravity
}

# Classify exoplanet
result = classifier.predict(input_data)
print(f"Classification: {result['prediction']}")
print(f"Confidence: {result['confidence']:.1%}")
```

### Batch Processing

```python
import pandas as pd
from backend_api import process_batch_classification

# Load candidate data
df = pd.read_csv('exoplanet_candidates.csv')

# Process all candidates
results = process_batch_classification(df)

# Save results
results.to_csv('classified_results.csv', index=False)
print(f"Processed {len(results)} candidates")
```

## Development

### Running in Development Mode

1. **Backend Development**:
```bash
# Start FastAPI with hot reload
uvicorn backend_api:app --reload --host 0.0.0.0 --port 8000
```

2. **Frontend Development**:
```bash
cd frontend
npm install
npm run dev
```

3. **Streamlit Development**:
```bash
streamlit run nasa_app_interface.py --server.port 8501
```

### Testing

```bash
# Run model validation
python nasa_clean_model.py

# Test API endpoints  
curl -X POST "http://localhost:8000/classify" \
     -H "Content-Type: application/json" \
     -d '{"koi_period": 365.25, "koi_prad": 1.0}'
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## NASA Space Apps Challenge 2025

### Challenge Requirements

- [x] **AI/ML Implementation**: Advanced ensemble models with 85.9% accuracy
- [x] **NASA Data Integration**: Kepler, K2, and TESS mission datasets  
- [x] **Real-world Application**: Production-ready full-stack system
- [x] **Innovation**: Smart model selection and feature engineering
- [x] **User Interface**: Interactive web application with batch processing
- [x] **Documentation**: Comprehensive setup and usage guides

### Technical Achievements

- **High Accuracy**: 85.9% classification accuracy with ensemble methods
- **Real-time Processing**: Sub-second prediction response times
- **Scalable Architecture**: Handles single and batch classifications
- **Professional Interface**: Modern React frontend with TypeScript
- **API-First Design**: RESTful backend with automatic documentation
- **Containerized Deployment**: Docker support for easy deployment

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- NASA Space Apps Challenge 2025
- NASA Exoplanet Archive
- Kepler/K2/TESS Mission Teams  
- Open source community contributors

## Live Demo

Experience the NASA Exoplanet Classifier: https://huggingface.co/spaces/ParthKoshti/Nasa-Exoplanet-Classifier

---

**Built for NASA Space Apps Challenge 2025 - "A World Away: Hunting for Exoplanets with AI"**

Discover worlds beyond our solar system with the power of artificial intelligence.
- [x] **Scalability**: Batch processing capabilities

- ** Democratizing Discovery**: Web interface for researchers
- ** Accelerating Science**: Automated classification pipeline
- ** Educational Value**: Interactive learning platform
- ** Research Tool**: Professional-grade analysis system

---

We welcome contributions to enhance the NASA Exoplanet Hunter!

- Additional NASA datasets integration
- Advanced AI model architectures
- UI/UX improvements
- Visualization enhancements
- New feature engineering approaches

---

This project is licensed under the MIT License.

---

- **NASA** for providing incredible exoplanet datasets
- **NASA Space Apps Challenge** for inspiring global innovation
- **Kepler/K2/TESS Teams** for groundbreaking space missions
- **Open Source Community** for amazing tools and libraries

---

- **NASA Space Apps**: [spaceappschallenge.org](https://www.spaceappschallenge.org)
- **NASA Exoplanet Archive**: [exoplanetarchive.ipac.caltech.edu](https://exoplanetarchive.ipac.caltech.edu)
- **GitHub Repository**: [github.com/Harshad2321/Exoplanet-Classifier-NASA-KOI-K2-TESS-](https://github.com/Harshad2321/Exoplanet-Classifier-NASA-KOI-K2-TESS-)

---

<div align="center">

**Advancing humanity's search for worlds beyond our solar system**

![NASA](https://img.shields.io/badge/NASA-Space%20Apps%20Challenge-blue)
![AI](https://img.shields.io/badge/AI-Exoplanet%20Hunter-orange)
![Mission](https://img.shields.io/badge/Mission-A%20World%20Away-green)

** Ready for NASA Space Apps Challenge 2025 Submission! **

</div>ation System
**NASA Space Apps Challenge 2025 - "A World Away: Hunting for Exoplanets with AI"**

> **Enterprise-grade machine learning solution** for classifying astronomical objects as confirmed exoplanets, planet candidates, or false positives using data from NASA's Kepler, K2, and TESS missions.

 **Achieving 69.19% accuracy with advanced ensemble methods** | **Production-ready architecture**

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20.0-orange.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![NASA](https://img.shields.io/badge/NASA-Space%20Apps%202025-red.svg)
![ML](https://img.shields.io/badge/ML-69.19%25%20Accuracy-brightgreen.svg)

---

- ** Modular Architecture**: Core system restructured with `core/`, `app/`, `api/` modules
- ** Performance Boost**: 90% faster predictions (2-3s → 0.1-0.3s) with model caching
- ** Memory Optimization**: Chunked data loading handles datasets larger than available RAM
- ** Modern UI**: Professional 4-page Streamlit interface with NASA branding
- ** Production Ready**: Comprehensive logging, error handling, and monitoring

| Model | Accuracy | Processing Time | Memory Usage |
|-------|----------|----------------|--------------|
| **WeightedEnsemble** | **69.19%** | **0.15s** | **Optimized** |
| RandomForest | 68.50% | 0.12s | Low |
| ExtraTrees | 68.20% | 0.18s | Medium |

---

```bash

git clone https://github.com/Harshad2321/Exoplanet-Classifier-NASA-KOI-K2-TESS-.git
cd Exoplanet-Classifier-NASA-KOI-K2-TESS-
pip install -r requirements.txt

streamlit run app/streamlit_app.py
```

```python
from core import get_prediction_api

predictor = get_prediction_api()

result = predictor.predict_single({
 'koi_period': 365.25,
 'koi_prad': 1.0,
 'koi_teq': 288,
 'koi_insol': 1.0
})

print(f"Classification: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")
```

```python
import pandas as pd
from core.prediction import BatchPredictor

df = pd.read_csv('your_exoplanet_data.csv')

batch_processor = BatchPredictor()

results = batch_processor.predict_batch(df)

results.to_csv('classified_exoplanets.csv', index=False)
```

---

- **Model Caching**: LRU cache with 3-model capacity for instant predictions
- **Async Processing**: Concurrent ensemble model execution
- **Error Recovery**: Graceful handling of model failures

- **Chunked Processing**: Handle datasets larger than available RAM
- **Smart Caching**: Disk-based caching with automatic cleanup
- **Memory Monitoring**: Automatic garbage collection at 80% threshold
- **Data Validation**: Schema validation and outlier detection

- **Environment-Aware**: Development/production settings
- **Feature Definitions**: NASA dataset column mappings
- **Comprehensive Logging**: Configurable levels and rotation

1. ** Single Prediction**: Individual exoplanet classification
2. ** Batch Analysis**: Multi-candidate processing with CSV upload
3. ** Model Comparison**: Side-by-side performance analysis
4. ** Data Explorer**: Interactive dataset analysis tools

- **Real-time Monitoring**: System health and model status
- **Interactive Visualizations**: Plotly charts and confidence gauges
- **Prediction History**: Track and review past classifications
- **Professional Branding**: NASA Space Apps Challenge theme

---

NASA's space missions have discovered thousands of potential exoplanets, but manual classification is time-intensive and prone to errors. Our AI system automates this process using advanced machine learning techniques to analyze stellar and planetary parameters.

- **Multi-Algorithm Ensemble**: Six advanced ML models with voting classifier
- **NASA Dataset Integration**: Kepler, K2, and TESS mission data
- **Real-time Classification**: Instant predictions with confidence scores
- **Interactive Dashboard**: Professional Streamlit web interface
- **Batch Processing**: Handle thousands of candidates simultaneously
- **Production Architecture**: Docker containers, REST APIs, comprehensive monitoring

---

 **Exoplanet Classification**: Multi-class prediction (CONFIRMED, CANDIDATE, FALSE POSITIVE)
 **NASA Data Integration**: Official Kepler/K2/TESS datasets
 **Advanced ML**: Ensemble methods achieving 69.19% accuracy
 **User Interface**: Professional web application for researchers
 **Real-world Impact**: Accelerate astronomical discovery workflows

- **AutoML Integration**: Automated feature engineering and model selection
- **Memory Optimization**: Handle massive astronomical datasets efficiently
- **Production Deployment**: Enterprise-grade architecture with monitoring
- **Educational Component**: Interactive data exploration for public engagement

---

```
WeightedEnsemble_L2: 69.19% CHAMPION
RandomForest: 68.50%
ExtraTreesClassifier: 68.20%
LightGBM: 67.85%
XGBoost: 67.40%
NeuralNet: 66.95%
```

- **Prediction Speed**: 0.1-0.3s per classification (90% improvement)
- **Memory Usage**: Optimized for large datasets (>10GB)
- **Batch Processing**: 1000+ candidates per minute
- **API Latency**: <100ms response time

---

```bash
Python 3.8+
RAM: 8GB+ recommended
Storage: 5GB free space
```

```bash

git clone https://github.com/Harshad2321/Exoplanet-Classifier-NASA-KOI-K2-TESS-.git
cd Exoplanet-Classifier-NASA-KOI-K2-TESS-

python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt

python -c "from core.prediction import PredictionAPI; PredictionAPI().load_models()"
```

```bash

python -c "from core import get_prediction_api; print(' Core system ready')"

streamlit run app/streamlit_app.py
```

---

```bash
streamlit run app/streamlit_app.py
```
- Navigate to: `http://localhost:8501`
- Use the 4-page interface for comprehensive analysis

```python
from core.prediction import PredictionAPI

api = PredictionAPI()

result = api.predict_single({
 'koi_period': 365.25,
 'koi_prad': 1.0,
 'koi_teq': 288,
 'koi_insol': 1.0,
 'koi_dor': 215.0,
 'koi_srad': 1.0
})

print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.1%}")
print(f"All Probabilities: {result['probabilities']}")
```

```python
import pandas as pd
from core.data_loader import DataLoader
from core.prediction import BatchPredictor

loader = DataLoader()
df = loader.load_csv('exoplanet_candidates.csv')

batch_predictor = BatchPredictor()
results = batch_predictor.predict_batch(df)

results.to_csv('classified_results.csv', index=False)
```

---

```
 Exoplanet-Classifier-NASA-KOI-K2-TESS-/
 core/
 __init__.py
 config.py
 prediction.py
 data_loader.py
 app/
 streamlit_app.py
 api/
 deployment/
 tests/
 data/
 models/
 notebooks/
 src/
 requirements.txt
```

---

- **Base Models**: 6 diverse algorithms (RandomForest, ExtraTrees, LightGBM, XGBoost, Neural Networks)
- **Ensemble Method**: Weighted voting with optimized model weights
- **Feature Engineering**: 37 NASA-defined astronomical parameters
- **Cross-Validation**: 5-fold stratified validation for robust evaluation

- **Source**: NASA Exoplanet Archive (official datasets)
- **Missions**: Kepler, K2, TESS combined data
- **Size**: 10,000+ classified objects
- **Classes**: CONFIRMED (exoplanets), CANDIDATE (potential), FALSE POSITIVE

Key astronomical parameters used by the model:
- **Orbital Characteristics**: Period, eccentricity, semi-major axis
- **Physical Properties**: Planetary radius, equilibrium temperature
- **Stellar Parameters**: Host star characteristics, insolation flux
- **Detection Metrics**: Signal-to-noise ratios, transit depths

---

```python
from src.enhanced_train import EnhancedModelTrainer

trainer = EnhancedModelTrainer(
 time_limit=7200,
 quality='high',
 ensemble_size=10
)

trainer.fit(train_data, target_column='koi_disposition')

trainer.save_model('custom_exoplanet_classifier.pkl')
```

```python
from src.explainability import ExplainablePredictor

explainer = ExplainablePredictor()

explanation = explainer.explain_prediction(sample_data)
print(explanation.feature_importance)

explainer.plot_shap_summary()
```

---

- [PHASE_1_COMPLETION_REPORT.md](./PHASE_1_COMPLETION_REPORT.md) - Detailed achievement summary
- [PROJECT_OPTIMIZATION_PLAN.md](./PROJECT_OPTIMIZATION_PLAN.md) - Development roadmap
- [models/README.md](./models/README.md) - Model documentation and performance

- **Core System**: Auto-generated docstrings and type hints
- **Web Interface**: Streamlit component documentation
- **Data Processing**: Feature engineering and preprocessing guides
- **Testing**: Unit test coverage and validation procedures

---

This project is designed for collaborative development during NASA Space Apps Challenge 2025.

```bash

git fork https://github.com/Harshad2321/Exoplanet-Classifier-NASA-KOI-K2-TESS-.git

git checkout -b feature/your-enhancement

python -m pytest tests/

git push origin feature/your-enhancement
```

- **Phase 2**: REST API implementation and Docker deployment
- **Advanced Models**: Deep learning architectures for improved accuracy
- **Data Integration**: Additional NASA mission datasets
- **Visualization**: Enhanced interactive plots and dashboards

---

MIT License - see [LICENSE](./LICENSE) for details

---

- **NASA Exoplanet Archive**: Primary dataset source
- **Kepler/K2 Mission**: Transit photometry data
- **TESS Mission**: All-sky survey observations

- **AutoML**: AutoGluon for automated machine learning
- **Visualization**: Streamlit, Plotly for interactive interfaces
- **ML Frameworks**: Scikit-learn, XGBoost, LightGBM
- **Data Processing**: Pandas, NumPy for efficient computation

---

**NASA Space Apps Challenge 2025 Team**
- **GitHub**: [Harshad2321/Exoplanet-Classifier-NASA-KOI-K2-TESS-](https://github.com/Harshad2321/Exoplanet-Classifier-NASA-KOI-K2-TESS-)
- **Challenge**: "A World Away: Hunting for Exoplanets with AI"

---

* Helping NASA discover new worlds, one prediction at a time!* 