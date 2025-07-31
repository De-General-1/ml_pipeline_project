#!/usr/bin/env python3

import os
import sys
import mlflow
import joblib

print("Testing MLflow and model loading...")

# Set MLflow tracking URI
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "file://" + os.path.abspath("./mlruns"))
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
print(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")

try:
    # Test loading preprocessing artifacts
    print("Loading preprocessing artifacts...")
    scaler = joblib.load("models/scaler.joblib")
    mlb = joblib.load("models/multilabel_binarizer.joblib")
    ohe_user_features = joblib.load("models/ohe_user.joblib")
    print("Preprocessing artifacts loaded successfully!")
    
    # Test loading model from registered model
    print("Loading model from registered model...")
    model = mlflow.sklearn.load_model("models:/MovieEngagementPredictor/Production")
    print("Model loaded successfully!")
    
    print("All tests passed!")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc() 