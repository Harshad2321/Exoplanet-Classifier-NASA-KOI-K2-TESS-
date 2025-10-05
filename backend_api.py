#!/usr/bin/env python3
"""
🚀 FastAPI Backend for NASA Exoplanet Classifier
Serves React frontend and provides classification API endpoints
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
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

class BatchClassificationResult(BaseModel):
    """Single row result in batch classification"""
    row_number: int
    classification: str
    confidence: float
    probabilities: Dict[str, float]
    input_data: Dict[str, float]

class BatchClassificationResponse(BaseModel):
    """Response model for batch classification"""
    total_rows: int
    successful: int
    failed: int
    results: list[BatchClassificationResult]
    errors: list[Dict[str, str]]
    
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

@app.post("/api/classify-batch", response_model=BatchClassificationResponse)
async def classify_batch(file: UploadFile = File(...)):
    """
    Classify multiple exoplanets from uploaded CSV file
    
    Expected CSV format:
    - First row: column headers matching ExoplanetInput field names
    - Subsequent rows: data values
    
    Example CSV:
    orbitalPeriod,stellarRadius,planetRadius,stellarMass,equilibriumTemperature,stellarTemperature,stellarAge,insolationFlux,distanceToStarRadius,rightAscension,declination,dispositionScore
    365.25,1.0,1.0,1.0,288.0,5778.0,4.6,1.0,1.0,0.0,0.0,0.9
    88.0,0.9,0.38,0.95,440.0,5700.0,4.5,6.67,0.39,45.0,15.0,0.85
    """
    
    if not models:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    # Validate file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    
    try:
        # Read CSV content
        contents = await file.read()
        csv_text = contents.decode('utf-8')
        
        # Parse CSV using pandas
        from io import StringIO
        df = pd.read_csv(StringIO(csv_text))
        
        logger.info(f"📊 Processing batch file: {file.filename} ({len(df)} rows)")
        
        results = []
        errors = []
        successful = 0
        failed = 0
        
        # Process each row
        for idx, row in df.iterrows():
            try:
                # Convert row to ExoplanetInput format
                row_data = {
                    'orbitalPeriod': float(row.get('orbitalPeriod', row.get('koi_period', 0))),
                    'stellarRadius': float(row.get('stellarRadius', row.get('koi_srad', 0))),
                    'planetRadius': float(row.get('planetRadius', row.get('koi_prad', 0))),
                    'stellarMass': float(row.get('stellarMass', row.get('koi_smass', 0))),
                    'equilibriumTemperature': float(row.get('equilibriumTemperature', row.get('koi_teq', 0))),
                    'stellarTemperature': float(row.get('stellarTemperature', row.get('koi_steff', 0))),
                    'stellarAge': float(row.get('stellarAge', row.get('koi_sage', 0))),
                    'insolationFlux': float(row.get('insolationFlux', row.get('koi_insol', 0))),
                    'distanceToStarRadius': float(row.get('distanceToStarRadius', row.get('koi_dor', 0))),
                    'rightAscension': float(row.get('rightAscension', row.get('ra', 0))),
                    'declination': float(row.get('declination', row.get('dec', 0))),
                    'dispositionScore': float(row.get('dispositionScore', row.get('koi_score', 0.5)))
                }
                
                # Create ExoplanetInput instance
                exoplanet = ExoplanetInput(**row_data)
                
                # Preprocess and classify
                X = preprocess_input(exoplanet)
                model = models.get('ensemble', list(models.values())[0])
                
                prediction_encoded = model.predict(X)[0]
                probabilities = model.predict_proba(X)[0]
                prediction = label_encoder.inverse_transform([prediction_encoded])[0]
                
                # Create probability dictionary
                prob_dict = {}
                for i, class_name in enumerate(label_encoder.classes_):
                    prob_dict[class_name] = float(probabilities[i])
                
                results.append(BatchClassificationResult(
                    row_number=int(idx) + 1,
                    classification=prediction,
                    confidence=float(max(probabilities)),
                    probabilities=prob_dict,
                    input_data=row_data
                ))
                
                successful += 1
                
            except Exception as row_error:
                failed += 1
                errors.append({
                    'row_number': int(idx) + 1,
                    'error': str(row_error)
                })
                logger.warning(f"⚠️ Row {idx + 1} failed: {row_error}")
        
        logger.info(f"✅ Batch processing complete: {successful} successful, {failed} failed")
        
        return BatchClassificationResponse(
            total_rows=len(df),
            successful=successful,
            failed=failed,
            results=results,
            errors=errors
        )
        
    except Exception as e:
        logger.error(f"❌ Batch classification error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")

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
    import os
    # Use port 7860 for Hugging Face Spaces, 8000 for local
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")