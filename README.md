# 🌌 Exoplanet Classifier - NASA Space Apps Challenge 2025

> **"A World Away: Hunting for Exoplanets with AI"**

An AI-powered machine learning solution for automatically classifying astronomical objects as confirmed exoplanets, planet candidates, or false positives using NASA's Kepler, K2, and TESS mission data.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![NASA](https://img.shields.io/badge/NASA-Space%20Apps%202025-red.svg)

---

## 🌟 Project Overview

NASA's space missions have discovered thousands of potential exoplanets, but manually classifying them is time-intensive. Our AI classifier automates this process by analyzing stellar and planetary parameters to distinguish between:
- **Confirmed Planets** 🪐
- **Planetary Candidates** 🌍
- **False Positives** ❌

### Key Features
- **Multi-Dataset Training**: Uses KOI (Kepler), K2, and TESS datasets
- **Advanced ML Pipeline**: From preprocessing to deployment
- **Interactive Web App**: Upload data and get instant predictions
- **Model Explainability**: Understand what drives classifications
- **Performance Metrics**: Comprehensive evaluation with visualizations

---

## 📊 Datasets

Our classifier leverages three major NASA exoplanet datasets:

| Dataset | Source | Classification Column | Records |
|---------|--------|----------------------|---------|
| **Kepler Objects of Interest (KOI)** | Kepler Mission | `koi_disposition` | ~10,000 |
| **K2 Planets and Candidates** | K2 Mission | `k2c_disp` | ~8,000 |
| **TESS Objects of Interest (TOI)** | TESS Mission | `tfopwg_disp` | ~6,000 |

**Key Features Used:**
- Orbital period, transit duration, planetary radius
- Stellar temperature, radius, and metallicity  
- Transit depth and signal-to-noise ratio

---

## 🛠️ Setup Instructions

### Prerequisites
- Python 3.8+
- pip package manager

### Quick Start
```bash
# Clone the repository
git clone https://github.com/yourusername/exoplanet-classifier.git
cd exoplanet-classifier

# Install dependencies
pip install -r requirements.txt

# Download datasets (automated)
python src/data_loader.py

# Train the model
python src/train_model.py

# Launch the web app
streamlit run app.py
```

### Manual Dataset Download
If automated download fails, manually download from [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/):
- Place CSV files in `data/raw/` directory
- Supported files: `koi.csv`, `k2candidates.csv`, `toi.csv`

---

## 📁 Project Structure

```
exoplanet-classifier/
├── 📁 data/
│   ├── raw/              # Original NASA datasets
│   ├── processed/        # Cleaned and feature-engineered data
│   └── splits/           # Train/validation/test splits
├── 📁 notebooks/
│   ├── 01_eda.ipynb          # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb # Data cleaning & feature engineering
│   └── 03_modeling.ipynb     # Model experiments & evaluation
├── 📁 src/
│   ├── data_loader.py        # Dataset download & loading utilities
│   ├── preprocessing.py      # Data cleaning & feature engineering
│   ├── train_model.py        # Model training pipeline
│   ├── evaluate_model.py     # Model evaluation & metrics
│   └── utils.py             # Helper functions
├── 📁 models/
│   ├── best_model.pkl       # Trained classifier
│   ├── scaler.pkl          # Feature scaler
│   └── feature_names.pkl   # Feature column names
├── app.py                   # Streamlit web application
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

---

## 🧠 Machine Learning Pipeline

### 1. Data Preprocessing
- **Missing Value Handling**: Smart imputation based on feature type
- **Feature Engineering**: Derived features from orbital mechanics
- **Normalization**: StandardScaler for numerical features
- **Class Balancing**: SMOTE for handling imbalanced classes

### 2. Model Architecture
- **Baseline Models**: Logistic Regression, Random Forest
- **Advanced Models**: XGBoost, LightGBM with hyperparameter tuning
- **Ensemble Method**: Voting classifier combining best performers
- **Cross-Validation**: 5-fold stratified CV for robust evaluation

### 3. Model Evaluation
- **Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Visualizations**: Confusion matrices, ROC curves, feature importance
- **Explainability**: SHAP values for model interpretability

---

## 🖥️ Web Application Features

### User Interface
- **Data Upload**: CSV file upload with validation
- **Manual Entry**: Form-based individual prediction
- **Batch Processing**: Multiple predictions at once

### Predictions & Insights
- **Classification Results**: Confidence scores for each class
- **Feature Importance**: Top factors influencing predictions
- **Model Performance**: Live accuracy metrics and visualizations
- **Data Exploration**: Interactive plots of training data

### Advanced Features
- **Model Retraining**: Upload new labeled data to improve model
- **Export Results**: Download predictions as CSV
- **API Endpoint**: RESTful API for programmatic access

---

## 📈 Performance Results

*Results will be updated after model training*

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | TBD | TBD | TBD | TBD | TBD |
| Random Forest | TBD | TBD | TBD | TBD | TBD |
| XGBoost | TBD | TBD | TBD | TBD | TBD |
| **Ensemble** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** |

---

## 🔬 Technical Approach

### Feature Engineering
- **Stellar Parameters**: Temperature, radius, metallicity normalization
- **Orbital Mechanics**: Period-radius relationships, habitable zone indicators
- **Transit Properties**: Depth ratios, duration consistency checks
- **Statistical Features**: Signal quality metrics, noise characterization

### Model Selection Criteria
- **Cross-validation performance** on held-out test set
- **Computational efficiency** for real-time predictions
- **Interpretability** for scientific validation
- **Robustness** across different datasets

---

## 🎯 NASA Space Apps Challenge Context

This project addresses the **"A World Away: Hunting for Exoplanets with AI"** challenge by:

1. **Leveraging Open Data**: Utilizing publicly available NASA datasets
2. **Automating Discovery**: Reducing manual classification effort
3. **Democratizing Access**: User-friendly interface for researchers
4. **Advancing Science**: Contributing to exoplanet research methodology

### Challenge Requirements Met
- ✅ Uses official NASA exoplanet datasets
- ✅ Implements machine learning classification
- ✅ Provides interactive web interface
- ✅ Includes model performance metrics
- ✅ Open-source and reproducible

---

## 🚀 Future Enhancements

- **Deep Learning**: CNN/RNN models for time-series light curves
- **Multi-Modal**: Combine tabular data with light curve images
- **Real-Time**: Stream processing for new TESS observations
- **Deployment**: Cloud hosting with CI/CD pipeline
- **Mobile App**: Smartphone interface for citizen science

---

## 👥 Team

*Add your team member information here*

- **Developer 1**: [Role] - [GitHub/LinkedIn]
- **Developer 2**: [Role] - [GitHub/LinkedIn]
- **Developer 3**: [Role] - [GitHub/LinkedIn]

---

## 📝 License

This project is open-source under the MIT License. See [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **NASA Exoplanet Archive** for providing open datasets
- **NASA Space Apps Challenge** for the inspiring challenge
- **Kepler/K2/TESS Teams** for their groundbreaking work
- **Open Source Community** for the amazing ML libraries

---

## 📞 Contact

For questions about this project, please open an issue or contact:
- Project Repository: [GitHub Link]
- NASA Space Apps Team Page: [Link]

**Made with ❤️ for NASA Space Apps Challenge 2025** 🚀