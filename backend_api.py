#!/usr/bin/env python3
"""
🚀 FastAPI Backend for NASA Exoplanet Classifier
Serves React frontend and provides classification API endpoints
"""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, Optional
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="NASA Exoplanet Classifier API",
    description="AI-powered exoplanet classification using NASA data",
    version="2.0.0"
)

# Configure CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models storage
MODELS_DIR = Path("nasa_models")
models = {}
scaler = None
imputer = None
label_encoder = None
metadata = None

class ExoplanetInput(BaseModel):
    """Input data model for exoplanet classification"""
    orbitalPeriod: float  # koi_period
    stellarRadius: float  # koi_srad
    planetRadius: float   # koi_prad
    stellarMass: float    # koi_smass
    equilibriumTemperature: float  # koi_teq
    stellarTemperature: float  # koi_steff
    stellarAge: float     # koi_sage
    insolationFlux: float  # koi_insol
    distanceToStarRadius: float  # koi_dor (frontend uses distanceToStarRadius)
    rightAscension: float  # ra
    declination: float    # dec
    dispositionScore: float  # koi_score

class ClassificationResponse(BaseModel):
    """Response model for classification results"""
    classification: str
    rationale: str  # Frontend expects 'rationale' not 'confidence'
    confidence: Optional[float] = None  # Optional for backward compatibility
    probabilities: Optional[Dict[str, float]] = None  # Optional
    model_used: Optional[str] = None  # Optional
    
class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    models_loaded: bool
    version: str

def load_models():
    """Load NASA AI models and preprocessing components"""
    global models, scaler, imputer, label_encoder, metadata
    
    try:
        logger.info("🚀 Loading NASA AI models...")
        
        # Load preprocessing components
        scaler_path = MODELS_DIR / 'nasa_scaler.pkl'
        imputer_path = MODELS_DIR / 'nasa_imputer.pkl'
        encoder_path = MODELS_DIR / 'nasa_label_encoder.pkl'
        metadata_path = MODELS_DIR / 'nasa_metadata.json'
        
        if all(path.exists() for path in [scaler_path, imputer_path, encoder_path, metadata_path]):
            scaler = joblib.load(scaler_path)
            imputer = joblib.load(imputer_path)
            label_encoder = joblib.load(encoder_path)
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            logger.info("✅ Preprocessing components loaded")
        else:
            logger.warning("⚠️  Some preprocessing files not found")
            return False
        
        # Load models
        model_files = {
            'random_forest': 'nasa_random_forest_model.pkl',
            'extra_trees': 'nasa_extra_trees_model.pkl',
            'ensemble': 'nasa_ensemble_model.pkl'
        }
        
        for model_name, filename in model_files.items():
            model_path = MODELS_DIR / filename
            if model_path.exists():
                models[model_name] = joblib.load(model_path)
                logger.info(f"✅ Loaded {model_name} model")
        
        if models:
            logger.info(f"🤖 Total models loaded: {len(models)}")
            return True
        
        return False
        
    except Exception as e:
        logger.error(f"❌ Error loading models: {e}")
        return False

def preprocess_input(data: ExoplanetInput) -> np.ndarray:
    """Preprocess input data for model prediction"""
    # Map React frontend field names to model feature names
    feature_mapping = {
        'koi_period': data.orbitalPeriod,
        'koi_prad': data.planetRadius,
        'koi_teq': data.equilibriumTemperature,
        'koi_insol': data.insolationFlux,
        'koi_srad': data.stellarRadius,
        'koi_smass': data.stellarMass,
        'koi_steff': data.stellarTemperature,
        'koi_sage': data.stellarAge,
        'koi_dor': data.distanceToStarRadius,  # Updated to match frontend
        'ra': data.rightAscension,
        'dec': data.declination,
        'koi_score': data.dispositionScore
    }
    
    # Create DataFrame
    df = pd.DataFrame([feature_mapping])
    
    # Ensure all required features are present
    required_features = metadata['feature_names']
    for feature in required_features:
        if feature not in df.columns:
            df[feature] = 0  # Default value for missing features
    
    # Select and order features correctly
    df = df[required_features]
    
    # Handle missing values (imputer expects original 12 features)
    X_imputed = imputer.transform(df)
    df_imputed = pd.DataFrame(X_imputed, columns=required_features, index=df.index)
    
    # Apply feature engineering
    df_engineered = engineer_features(df_imputed)
    
    # Scale features
    X_scaled = scaler.transform(df_engineered)
    
    return X_scaled

def engineer_features(X: pd.DataFrame) -> pd.DataFrame:
    """Engineer additional features (same as training)"""
    X_eng = X.copy()
    
    # 1. planet_mass_proxy
    if 'koi_prad' in X_eng.columns:
        X_eng['planet_mass_proxy'] = X_eng['koi_prad'] ** 2.06
    
    # 2. temp_ratio
    if 'koi_teq' in X_eng.columns and 'koi_steff' in X_eng.columns:
        X_eng['temp_ratio'] = X_eng['koi_teq'] / X_eng['koi_steff']
    
    # 3. orbital_velocity
    if all(col in X_eng.columns for col in ['koi_period', 'koi_dor', 'koi_srad']):
        X_eng['orbital_velocity'] = (2 * np.pi * X_eng['koi_dor'] * X_eng['koi_srad']) / X_eng['koi_period']
    
    # 4. habitable_zone
    if 'koi_teq' in X_eng.columns:
        X_eng['habitable_zone'] = ((X_eng['koi_teq'] >= 200) & (X_eng['koi_teq'] <= 400)).astype(int)
    
    # 5. transit_depth
    if 'koi_prad' in X_eng.columns and 'koi_srad' in X_eng.columns:
        X_eng['transit_depth'] = (X_eng['koi_prad'] / (109 * X_eng['koi_srad'])) ** 2
    
    # Ensure columns are in the exact training order
    expected_columns = [
        'koi_period', 'koi_prad', 'koi_teq', 'koi_insol', 'koi_srad', 'koi_smass',
        'koi_steff', 'koi_sage', 'koi_dor', 'ra', 'dec', 'koi_score',
        'planet_mass_proxy', 'temp_ratio', 'orbital_velocity', 'habitable_zone', 'transit_depth'
    ]
    
    X_eng = X_eng.reindex(columns=expected_columns, fill_value=0)
    
    return X_eng

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    logger.info("🚀 Starting NASA Exoplanet Classifier API...")
    success = load_models()
    if success:
        logger.info("✅ API ready to classify exoplanets!")
    else:
        logger.error("❌ Failed to load models. API may not function correctly.")

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if models else "unhealthy",
        "models_loaded": bool(models),
        "version": "2.0.0"
    }

@app.post("/api/classify", response_model=ClassificationResponse)
async def classify_exoplanet(data: ExoplanetInput):
    """
    Classify an exoplanet based on astronomical parameters
    """
    try:
        if not models:
            raise HTTPException(status_code=503, detail="Models not loaded")
        
        logger.info("🔭 Classifying exoplanet...")
        
        # Preprocess input
        X = preprocess_input(data)
        
        # Use ensemble model (best performance)
        model = models.get('ensemble', list(models.values())[0])
        
        # Make prediction
        prediction_encoded = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]
        
        # Decode prediction
        prediction = label_encoder.inverse_transform([prediction_encoded])[0]
        
        # Create probability dictionary
        prob_dict = {}
        for i, class_name in enumerate(label_encoder.classes_):
            prob_dict[class_name] = float(probabilities[i])
        
        logger.info(f"✅ Classification: {prediction} (confidence: {max(probabilities):.1%})")
        
        # Create rationale based on classification
        confidence_pct = float(max(probabilities)) * 100
        if prediction == "CONFIRMED":
            rationale = f"High confidence ({confidence_pct:.1f}%) confirmed exoplanet based on NASA AI analysis of orbital and stellar parameters."
        elif prediction == "CANDIDATE":
            rationale = f"Moderate confidence ({confidence_pct:.1f}%) exoplanet candidate requiring further observation and verification."
        else:
            rationale = f"High confidence ({confidence_pct:.1f}%) this is likely a false positive, not a true exoplanet detection."
        
        return {
            "classification": prediction,
            "rationale": rationale,
            "confidence": float(max(probabilities)),
            "probabilities": prob_dict,
            "model_used": "Ensemble AI Model"
        }
        
    except Exception as e:
        logger.error(f"❌ Classification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/models")
async def list_models():
    """List available AI models"""
    if not models:
        return {"models": [], "status": "No models loaded"}
    
    return {
        "models": list(models.keys()),
        "status": "ready",
        "metadata": metadata
    }

# Serve React frontend static files
@app.get("/")
async def serve_frontend():
    """Serve the React frontend"""
    frontend_path = Path("frontend/dist/index.html")
    if frontend_path.exists():
        return FileResponse(frontend_path)
    else:
        return {
            "message": "NASA Exoplanet Classifier API",
            "status": "running",
            "docs": "/docs",
            "frontend": "Build the React app first: cd frontend && npm run build"
        }

# Mount static files (after building React app)
try:
    app.mount("/assets", StaticFiles(directory="frontend/dist/assets"), name="assets")
    logger.info("✅ Frontend static files mounted")
except Exception as e:
    logger.warning(f"⚠️  Could not mount frontend static files: {e}")
    logger.info("💡 Build the frontend first: cd frontend && npm install && npm run build")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")