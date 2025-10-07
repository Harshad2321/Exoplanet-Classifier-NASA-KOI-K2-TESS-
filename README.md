# NASA Exoplanet Classifier

An advanced AI-powered exoplanet classification system built for the NASA Space Apps Challenge 2025. This project leverages machine learning algorithms to analyze and classify exoplanet candidates from NASA's KOI (Kepler Objects of Interest), K2, and TESS missions data.

## Live Demo

**Try the live application:** [https://huggingface.co/spaces/ParthKoshti/Nasa-Exoplanet-Classifier](https://huggingface.co/spaces/ParthKoshti/Nasa-Exoplanet-Classifier)

## Overview

The NASA Exoplanet Classifier is a full-stack application that combines sophisticated machine learning models with an intuitive user interface to classify exoplanet candidates into three categories:
- **CANDIDATE**: Potential exoplanet requiring further investigation
- **CONFIRMED**: Verified exoplanet with high confidence
- **FALSE_POSITIVE**: Object ruled out as an exoplanet

## Features

### Machine Learning Models
- **Ensemble Learning**: Combines multiple algorithms for improved accuracy
- **Random Forest**: Robust classification with feature importance analysis
- **Extra Trees**: Enhanced decision tree ensemble for complex patterns
- **Smart AI Model Selection**: Automatically selects the best model for each prediction
- **Advanced Data Processing**: Automated feature scaling, imputation, and encoding

### Web Application
- **React + TypeScript Frontend**: Modern, responsive user interface
- **FastAPI Backend**: High-performance API with automatic documentation
- **Streamlit Interface**: Alternative interface for data scientists
- **Real-time Predictions**: Instant classification results
- **Batch Processing**: Upload and process multiple data points
- **Interactive Visualizations**: Plotly-powered charts and graphs

### Key Capabilities
- Single and batch exoplanet classification
- Feature importance visualization
- Model performance metrics
- Data preprocessing and validation
- Confidence scoring for predictions
- Export results in multiple formats

## Technology Stack

### Backend
- **Python 3.8+**
- **FastAPI**: Modern web framework for APIs
- **Scikit-learn**: Machine learning library
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Joblib**: Model serialization
- **Streamlit**: Data app framework

### Frontend
- **React 19**: JavaScript library for user interfaces
- **TypeScript**: Type-safe JavaScript
- **Vite**: Fast build tool and development server
- **GSAP**: Animation library
- **OGL**: WebGL library for 3D graphics

### Machine Learning
- **Ensemble Methods**: Random Forest, Extra Trees
- **Data Preprocessing**: StandardScaler, SimpleImputer
- **Feature Engineering**: Label encoding, feature selection
- **Model Evaluation**: Cross-validation, performance metrics

## Installation

### Prerequisites
- Python 3.8 or higher
- Node.js 16 or higher
- npm or yarn package manager

### Backend Setup

1. Clone the repository:
```bash
git clone https://github.com/Harshad2321/Exoplanet-Classifier-NASA-KOI-K2-TESS-.git
cd Exoplanet-Classifier-NASA-KOI-K2-TESS-
```

2. Create a virtual environment:
```bash
python -m venv nasa_env
source nasa_env/bin/activate  # On Windows: nasa_env\Scripts\activate
```

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install Node.js dependencies:
```bash
npm install
```

3. Build the frontend:
```bash
npm run build
```

## Usage

### Quick Start

1. **Run the full-stack application**:
```bash
python start_app.py
```

2. **Access the application**:
   - Frontend: http://localhost:8000
   - API Documentation: http://localhost:8000/docs
   - Alternative Streamlit Interface: http://localhost:8501

### Alternative Launch Methods

**Streamlit Interface**:
```bash
streamlit run nasa_app_interface.py
```

**FastAPI Backend Only**:
```bash
python backend_api.py
```

**Frontend Development Server**:
```bash
cd frontend
npm run dev
```

### Docker Deployment

Build and run with Docker:
```bash
docker build -t nasa-exoplanet-classifier .
docker run -p 8000:8000 nasa-exoplanet-classifier
```

## API Endpoints

### Classification
- `POST /classify/single` - Classify a single exoplanet candidate
- `POST /classify/batch` - Classify multiple candidates
- `GET /models/info` - Get model information and metrics

### Data Requirements

The classifier expects the following 12 features for each exoplanet candidate:

| Feature | Description | Unit |
|---------|-------------|------|
| koi_period | Orbital period | days |
| koi_prad | Planet radius | Earth radii |
| koi_teq | Equilibrium temperature | Kelvin |
| koi_insol | Stellar insolation | Earth flux |
| koi_srad | Stellar radius | Solar radii |
| koi_smass | Stellar mass | Solar masses |
| koi_steff | Stellar effective temperature | Kelvin |
| koi_sage | Stellar age | billion years |
| koi_dor | Planet-star distance ratio | - |
| ra | Right ascension | degrees |
| dec | Declination | degrees |
| koi_score | KOI score | - |

## Model Performance

The ensemble model achieves high accuracy across all classification categories:
- **Overall Accuracy**: > 95%
- **Precision**: High precision for all classes
- **Recall**: Balanced recall across categories
- **F1-Score**: Optimized for scientific applications

## Project Structure

```
├── backend_api.py              # FastAPI backend server
├── nasa_app_interface.py       # Streamlit application
├── nasa_smart_classifier.py    # Smart AI model selection
├── nasa_clean_model.py         # Data preprocessing utilities
├── start_app.py               # Application launcher
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Docker configuration
├── frontend/                  # React frontend application
│   ├── src/
│   ├── package.json
│   └── vite.config.ts
└── nasa_models/              # Trained ML models
    ├── nasa_ensemble_model.pkl
    ├── nasa_random_forest_model.pkl
    ├── nasa_extra_trees_model.pkl
    ├── nasa_scaler.pkl
    ├── nasa_imputer.pkl
    ├── nasa_label_encoder.pkl
    └── nasa_metadata.json
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Data Sources

This project utilizes data from:
- **NASA Kepler Mission**: KOI catalog
- **K2 Mission**: Extended Kepler observations
- **TESS Mission**: Transiting Exoplanet Survey Satellite
- **NASA Exoplanet Archive**: Comprehensive exoplanet database

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- NASA Space Apps Challenge 2025
- NASA Kepler, K2, and TESS mission teams
- NASA Exoplanet Archive
- Open source machine learning community

## Authors

- **Harshad Patel** - [@Harshad2321](https://github.com/Harshad2321)
- **Parth Koshti** - HuggingFace Space Deployment

## Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Contact the development team
- Check the documentation in the `/docs` directory

---

**Built with passion for space exploration and artificial intelligence**