# NASA Space Apps Challenge 2025 - Exoplanet Classifier

## Model Information
- **Challenge**: A World Away: Hunting for Exoplanets with AI
- **Best Model**: random_forest
- **Accuracy**: 87.7%
- **Classes**: CONFIRMED, CANDIDATE, FALSE_POSITIVE

## Performance Summary
- **Random Forest**: 87.7% accuracy
- **Extra Trees**: 83.9% accuracy  
- **Ensemble**: 86.4% accuracy

## Features Used (23 total)
Key features include orbital period, planet radius, temperature, stellar properties, and detection confidence indicators.

## Usage
Load the model using the deployment interface:
```bash
streamlit run deploy_app.py
```

## NASA Space Apps Challenge 2025
This model helps astronomers classify objects detected by NASA's Kepler, K2, and TESS missions.

**Impact**: Accelerate exoplanet discovery and help identify potentially habitable worlds!