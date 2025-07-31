from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import boto3
import pickle
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import os
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Movie Engagement Prediction API", version="1.0.0")

# Global variables for model artifacts
model = None
scaler = None
mlb = None
ohe_user = None
feature_columns = None
last_model_update = None

class PredictionRequest(BaseModel):
    user_id: int
    movie_id: int
    age: int
    gender: str
    occupation: str
    genres: List[str]
    release_year: int

class PredictionResponse(BaseModel):
    user_id: int
    movie_id: int
    engagement_probability: float
    engagement_prediction: int
    model_version: str
    prediction_timestamp: str

def load_model_from_s3():
    """Load model and preprocessing artifacts from S3"""
    global model, scaler, mlb, ohe_user, feature_columns, last_model_update
    
    try:
        s3_bucket = os.getenv('S3_BUCKET', 'phase3-mlops-source-bucket-degen-1')
        model_path = os.getenv('MODEL_PATH', 'models/random_forest')
        
        # Use IAM role in production, profile for local development
        if os.getenv('AWS_EXECUTION_ENV'):
            s3_client = boto3.client('s3')
        else:
            session = boto3.Session(profile_name='degen-mlops')
            s3_client = session.client('s3')
        
        # Download model artifacts
        artifacts = ['model.pkl', 'scaler.pkl', 'multilabel_binarizer.pkl', 'ohe_user.pkl']
        
        for artifact in artifacts:
            s3_key = f"{model_path}/{artifact}"
            local_path = f"/tmp/{artifact}"
            
            logger.info(f"Downloading {s3_key} from S3...")
            s3_client.download_file(s3_bucket, s3_key, local_path)
        
        # Load artifacts
        with open('/tmp/model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        with open('/tmp/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        with open('/tmp/multilabel_binarizer.pkl', 'rb') as f:
            mlb = pickle.load(f)
        
        with open('/tmp/ohe_user.pkl', 'rb') as f:
            ohe_user = pickle.load(f)
        
        # Define feature columns (same as training)
        numerical_features = ['ReleaseYear']
        categorical_features_genres = list(mlb.classes_)
        categorical_features_users = list(ohe_user.get_feature_names_out(['Gender', 'Age', 'Occupation']))
        feature_columns = numerical_features + categorical_features_genres + categorical_features_users
        
        last_model_update = datetime.now()
        logger.info("Model loaded successfully from S3")
        
    except Exception as e:
        logger.error(f"Error loading model from S3: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

def preprocess_input(request: PredictionRequest) -> pd.DataFrame:
    """Preprocess input data to match training format"""
    try:
        # Create base dataframe
        data = {
            'ReleaseYear': [request.release_year],
            'Gender': [request.gender],
            'Age': [request.age],
            'Occupation': [request.occupation]
        }
        
        df = pd.DataFrame(data)
        
        # Process genres
        genre_features = mlb.transform([request.genres])
        genre_df = pd.DataFrame(genre_features, columns=mlb.classes_)
        
        # Process user features
        user_features = ohe_user.transform(df[['Gender', 'Age', 'Occupation']])
        user_df = pd.DataFrame(user_features, columns=ohe_user.get_feature_names_out(['Gender', 'Age', 'Occupation']))
        
        # Combine all features
        result_df = pd.concat([
            df[['ReleaseYear']],
            genre_df,
            user_df
        ], axis=1)
        
        # Ensure all feature columns are present
        for col in feature_columns:
            if col not in result_df.columns:
                result_df[col] = 0
        
        # Reorder columns to match training
        result_df = result_df[feature_columns]
        
        # Scale numerical features
        result_df['ReleaseYear'] = scaler.transform(result_df[['ReleaseYear']])
        
        return result_df
        
    except Exception as e:
        logger.error(f"Error preprocessing input: {e}")
        raise HTTPException(status_code=400, detail=f"Preprocessing error: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model_from_s3()

@app.get("/")
async def root():
    return {"message": "Movie Engagement Prediction API", "status": "running"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    model_status = "loaded" if model is not None else "not_loaded"
    return {
        "status": "healthy",
        "model_status": model_status,
        "last_model_update": last_model_update.isoformat() if last_model_update else None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make engagement prediction"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Preprocess input
        input_df = preprocess_input(request)
        
        # Make prediction
        prediction_proba = model.predict_proba(input_df)[0]
        engagement_probability = float(prediction_proba[1])  # Probability of engagement (class 1)
        engagement_prediction = int(model.predict(input_df)[0])
        
        return PredictionResponse(
            user_id=request.user_id,
            movie_id=request.movie_id,
            engagement_probability=engagement_probability,
            engagement_prediction=engagement_prediction,
            model_version="random_forest_v1",
            prediction_timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/reload-model")
async def reload_model():
    """Reload model from S3"""
    try:
        load_model_from_s3()
        return {"message": "Model reloaded successfully", "timestamp": last_model_update.isoformat()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload model: {str(e)}")

@app.get("/model-info")
async def model_info():
    """Get model information"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": type(model).__name__,
        "feature_count": len(feature_columns),
        "last_update": last_model_update.isoformat() if last_model_update else None,
        "s3_bucket": os.getenv('S3_BUCKET'),
        "model_path": os.getenv('MODEL_PATH')
    }