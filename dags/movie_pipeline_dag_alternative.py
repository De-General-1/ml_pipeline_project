from __future__ import annotations

import pendulum
import os
import pandas as pd # Import pandas as it's used in the python_callable
import numpy as np  # Import numpy if it's used
import mlflow       # Import mlflow if it's used
import joblib       # Import joblib if it's used

from airflow.models.dag import DAG
from airflow.operators.python import PythonOperator

# Alternative import approach - import the module directly
import importlib.util
import sys

def load_train_module():
    """Load the train module dynamically"""
    train_path = '/opt/airflow/scripts/train.py'
    spec = importlib.util.spec_from_file_location("train", train_path)
    train_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_module)
    return train_module

# Load the train module
try:
    train_module = load_train_module()
    _load_and_initial_preprocess_data = train_module._load_and_initial_preprocess_data
    _train_and_evaluate_model_flow = train_module._train_and_evaluate_model_flow
    print("✅ Successfully loaded train module")
except Exception as e:
    print(f"❌ Error loading train module: {e}")
    raise

# Define a Python callable to orchestrate the ML pipeline steps
def ml_pipeline_callable():
    """
    Orchestrates the data loading, preprocessing, and model training/evaluation.
    This function will be called by Airflow's PythonOperator.
    """
    print("Starting ML pipeline callable...")
    
    # Set MLflow tracking URI for this process, if not already set by Airflow's environment
    # Ensure it's pointing to the shared volume with proper permissions
    mlruns_path = "/opt/airflow/mlruns"
    tracking_uri = f"file://{mlruns_path}"
    os.environ['MLFLOW_TRACKING_URI'] = tracking_uri
    mlflow.set_tracking_uri(tracking_uri)
    print(f"MLflow tracking set to: {tracking_uri}")
    
    # Ensure the mlruns directory exists and has proper permissions
    os.makedirs(mlruns_path, exist_ok=True)

    # 1. Load and Initial Preprocessing
    # This step returns the fully merged and preprocessed DataFrame (X), target (y),
    # and the fitted preprocessors (mlb, ohe_user_features).
    X, y, mlb, ohe_user_features = _load_and_initial_preprocess_data()
    print("Data loaded and initially preprocessed.")

    # 2. Train and Evaluate Models
    # Pass the results from the previous step to the training function
    # The training function internally handles splitting, scaling, and MLflow logging.
    _train_and_evaluate_model_flow(
        X=X, 
        y=y, 
        mlb=mlb, 
        ohe_user_features=ohe_user_features,
        test_size=0.2,
        random_state=42
    )
    print("Model training and evaluation complete.")

with DAG(
    dag_id="movie_engagement_ml_pipeline_alternative",
    start_date=pendulum.datetime(2023, 1, 1, tz="UTC"),
    catchup=False,
    schedule=None, # Set a schedule like "@daily" or "0 0 * * *" for daily run
    tags=["mlops", "movie_recommendation"],
    doc_md="""
    ### Movie Engagement ML Pipeline (Alternative Import Method)
    An Airflow DAG to orchestrate the MovieLens 100K dataset download, preprocessing,
    and model training (Logistic Regression & RandomForestClassifier) using MLflow for tracking.
    Uses dynamic module loading to avoid import issues.
    """
) as dag:
    # Task to run the entire ML pipeline flow
    run_ml_pipeline = PythonOperator(
        task_id="run_full_ml_pipeline",
        python_callable=ml_pipeline_callable,
    ) 