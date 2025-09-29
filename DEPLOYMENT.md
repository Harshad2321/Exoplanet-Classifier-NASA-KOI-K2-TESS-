# 🚀 NASA Space Apps Challenge 2025: Exoplanet Hunter AI

**Challenge:** "A World Away: Hunting for Exoplanets with AI"

## 🏆 Competition Entry

A professional AI-powered exoplanet classification system built for the NASA Space Apps Challenge 2025, featuring an ensemble machine learning model with 85.9% accuracy and a production-ready Streamlit web interface.

## ✨ Features

### 🤖 AI Classification Engine
- **Ensemble Model**: Random Forest + Extra Trees (85.9% accuracy)
- **Real-time Predictions**: Single and batch exoplanet classification
- **Feature Engineering**: 17 astronomical parameters with domain knowledge
- **Robust Pipeline**: Imputation → Engineering → Scaling → Prediction

### 🌐 Web Interface
- **Professional Design**: NASA-themed interface with mission branding
- **Multi-tab Layout**: 
  - 🔭 Single Classification
  - 📊 Batch Analysis  
  - 📈 Mission Dashboard
  - 🎓 About Challenge
- **Interactive Visualizations**: Real-time charts and statistics
- **Error Handling**: Robust CSV processing with malformed data recovery

### 🛰️ NASA Mission Integration
- **Kepler Mission**: Historical exoplanet discoveries
- **K2 Mission**: Extended observations
- **TESS Mission**: All-sky survey data
- **Real Parameters**: Authentic astronomical measurements

## 🚀 Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/Harshad2321/Exoplanet-Classifier-NASA-KOI-K2-TESS-.git
cd Exoplanet-Classifier-NASA-KOI-K2-TESS-

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Training the Model (Optional)
```bash
# Train new models (takes ~2-3 minutes)
python nasa_clean_model.py
```

### Launch the Web Interface
```bash
# Start the NASA Exoplanet Hunter
streamlit run nasa_app_interface.py
```

Open your browser to `http://localhost:8501` 🌌

## 📊 Model Performance

| Model Component | Accuracy | Purpose |
|----------------|----------|---------|
| **Ensemble** | **85.9%** | Final predictions |
| Random Forest | 84.2% | Robust classification |
| Extra Trees | 83.7% | Variance reduction |

### Classification Classes
- **CONFIRMED**: Verified exoplanet (🪐)
- **CANDIDATE**: Potential exoplanet (🔍)  
- **FALSE_POSITIVE**: Not an exoplanet (❌)

## 🔬 Technical Architecture

### Data Pipeline
```
Raw KOI Data → Imputation → Feature Engineering → Scaling → AI Model → Prediction
     (12)         (12)            (17)           (17)      (3 classes)
```

### Key Features (17 total)
**Original (12):** Period, Radius, Temperature, Insolation, Stellar params, Coordinates, Score

**Engineered (5):** Mass proxy, Temperature ratio, Orbital velocity, Habitable zone, Transit depth

### Files Structure
```
├── nasa_clean_model.py      # AI training pipeline
├── nasa_app_interface.py    # Streamlit web app
├── nasa_models/            # Trained AI models
│   ├── nasa_ensemble_model.pkl
│   ├── nasa_scaler.pkl
│   ├── nasa_imputer.pkl
│   └── ...
├── sample_koi_data.csv     # Test dataset
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## 🎯 Usage Examples

### Single Exoplanet Classification
```python
# Earth-like parameters
input_data = {
    'koi_period': 365.25,     # Orbital period (days)
    'koi_prad': 1.0,          # Planet radius (Earth radii)
    'koi_teq': 288.0,         # Temperature (K)
    'koi_insol': 1.0,         # Insolation flux
    # ... other parameters
}

# Result: CONFIRMED exoplanet (65.4% confidence)
```

### Batch Processing
Upload CSV files with multiple KOI observations for bulk analysis with automatic error handling.

## 🏆 NASA Space Apps Challenge Compliance

### Challenge Requirements ✅
- **Theme**: "A World Away: Hunting for Exoplanets with AI"
- **AI Integration**: Advanced ensemble machine learning
- **Real Data**: Kepler/K2/TESS mission parameters
- **User Interface**: Professional web application
- **Documentation**: Comprehensive project docs
- **Open Source**: MIT License

### Innovation Highlights
- **Domain-Aware AI**: Astronomical feature engineering
- **Production Ready**: Robust error handling & validation
- **Educational**: Interactive learning about exoplanets
- **Scalable**: Handles single predictions to bulk analysis

## 🌌 Scientific Background

### Exoplanet Detection Methods
- **Transit Method**: Planet dims star light when passing in front
- **Radial Velocity**: Star wobbles due to planet's gravity
- **Direct Imaging**: Directly observing planet light

### Habitability Assessment
The system automatically evaluates:
- **Temperature Range**: 200K - 400K (habitable zone)
- **Planet Size**: Earth-like to Super-Earth
- **Orbital Characteristics**: Stable, long-term orbits

## 🔧 Development

### Model Retraining
```bash
# Modify parameters in nasa_clean_model.py
python nasa_clean_model.py
```

### Adding Features
1. Update `_engineer_features()` in both training and prediction
2. Ensure feature order consistency
3. Retrain models

### Deployment
The system is ready for:
- **Local deployment**: Streamlit server
- **Cloud deployment**: Heroku, AWS, GCP
- **Docker containers**: Scalable deployment

## 🤝 Contributing

This project is open for collaboration! Areas for enhancement:
- Additional ML algorithms
- More astronomical features
- Enhanced visualizations
- Performance optimizations

## 📜 License

MIT License - Feel free to use for educational and research purposes.

## 🙏 Acknowledgments

- **NASA**: For providing open exoplanet data
- **Kepler/K2/TESS Teams**: For groundbreaking discoveries
- **Space Apps Challenge**: For inspiring innovation
- **Open Source Community**: For amazing tools and libraries

---

**🌟 Built with passion for space exploration and AI innovation! 🌟**

*Ready to discover new worlds? Launch the NASA Exoplanet Hunter and start your journey! 🚀*