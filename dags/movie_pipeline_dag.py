from __future__ import annotations

import pendulum
import os
import pandas as pd
import numpy as np
import mlflow
import joblib

# Import the functions directly from your scripts/train.py module
import sys
# Ensure /opt/airflow/scripts is on the Python path for direct imports
# This line is good to keep in case environment variables aren't fully propagated,
# though docker-compose.yaml's PYTHONPATH is the primary mechanism.
sys.path.insert(0, '/opt/airflow/scripts')

try:
    from train import (
        _download_and_extract_data, # We'll expose this now
        _load_and_initial_preprocess_data, 
        _train_and_evaluate_model_flow,
        MOVIELENS_100K_URL, # Bring in constants
        DOWNLOAD_DIR,
        DATA_ROOT_DIR,
        MLFLOW_EXPERIMENT_NAME, # Bring in MLflow constants
        MLFLOW_REGISTERED_MODEL_NAME
    )
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Current sys.path: {sys.path}")
    print(f"Scripts directory exists: {os.path.exists('/opt/airflow/scripts')}")
    print(f"Scripts directory contents: {os.listdir('/opt/airflow/scripts') if os.path.exists('/opt/airflow/scripts') else 'Directory not found'}")
    raise

from airflow.models.dag import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.email import send_email_smtp # For task failure notifications

# --- Define callable functions for individual tasks ---

def _download_data_callable():
    """Airflow callable to download and extract the dataset."""
    print("Starting data download and extraction...")
    _download_and_extract_data(MOVIELENS_100K_URL, DOWNLOAD_DIR, DATA_ROOT_DIR)
    print("Data download and extraction complete.")

def _preprocess_data_callable():
    """Airflow callable to load and preprocess data."""
    print("Starting data loading and initial preprocessing...")
    X, y, mlb, ohe_user_features = _load_and_initial_preprocess_data()
    print("Data loaded and initially preprocessed.")
    
    # Pass data between tasks using XCom (Cross-Communication)
    # For large datasets, this isn't ideal, but for small ones or
    # passing metadata/paths, it's common.
    # Here, we'll simulate passing by writing to a shared location or
    # just acknowledging it's processed. For actual data passing in a real pipeline,
    # you'd save X and y to Parquet/CSV in /opt/airflow/data and pass the paths.
    
    # For simplicity and given our functions' structure, _train_and_evaluate_model_flow
    # already calls _load_and_initial_preprocess_data internally.
    # We will adjust this to truly separate tasks by making preprocess save data,
    # and train load it.

def _train_models_callable():
    """Airflow callable to train and evaluate models."""
    print("Starting model training and evaluation...")
    
    # Ensure MLflow tracking URI is set for this process
    mlruns_path = "/opt/airflow/mlruns"
    tracking_uri = f"file://{mlruns_path}"
    os.environ['MLFLOW_TRACKING_URI'] = tracking_uri
    mlflow.set_tracking_uri(tracking_uri)
    print(f"MLflow tracking set to: {tracking_uri}")
    
    # Ensure the mlruns directory exists and has proper permissions
    os.makedirs(mlruns_path, exist_ok=True)
    
    # Re-load data for training from the place it was prepared (or just call the full flow)
    # For a real pipeline, _preprocess_data_callable would save X, y, mlb, ohe_user_features
    # to /opt/airflow/data or /opt/airflow/models, and _train_models_callable would load them.
    # Given the current structure of `_train_and_evaluate_model_flow` which calls `_load_and_initial_preprocess_data`
    # internally, we'll keep the simplicity for now.
    
    # To truly separate them, _load_and_initial_preprocess_data would need to save X, y, mlb, ohe_user_features
    # to disk (e.g., as parquet files and joblib files in /opt/airflow/data or /opt/airflow/models).
    # Then, this _train_models_callable would load those saved files.
    
    # For now, we'll call the flow that encapsulates both load and train for simplicity of transition
    # to truly separate steps.
    X, y, mlb, ohe_user_features = _load_and_initial_preprocess_data()
    _train_and_evaluate_model_flow(
        X=X, 
        y=y, 
        mlb=mlb, 
        ohe_user_features=ohe_user_features,
        test_size=0.2,
        random_state=42
    )
    print("Model training and evaluation complete.")

# --- Airflow DAG Definition ---
with DAG(
    dag_id="movie_engagement_ml_pipeline",
    start_date=pendulum.datetime(2023, 1, 1, tz="UTC"),
    catchup=False,
    schedule=None, # Set a schedule like "@daily" or "0 0 * * *" for daily run
    tags=["mlops", "movie_recommendation"],
    doc_md="""
    ### Movie Engagement ML Pipeline
    An Airflow DAG to orchestrate the MovieLens 100K dataset download, preprocessing,
    and model training (Logistic Regression & RandomForestClassifier) using MLflow for tracking.
    """,
    # Add email on failure for production scenarios
    # on_failure_callback=lambda context: send_email_smtp(
    #     to='your_email@example.com',
    #     subject=f"Airflow DAG Failure: {context['dag'].dag_id}",
    #     html_content=f"Task {context['task_instance'].task_id} failed on {context['ds']}"
    # )
) as dag:
    # 1. Task for data downloading and extraction
    download_data = PythonOperator(
        task_id="download_data",
        python_callable=_download_data_callable,
    )

    # 2. Task for data preprocessing
    # IMPORTANT: Currently, _train_and_evaluate_model_flow calls _load_and_initial_preprocess_data.
    # For a truly modular pipeline, _load_and_initial_preprocess_data should save its outputs (X, y, mlb, ohe)
    # to disk (e.g., in /opt/airflow/data or /opt/airflow/models), and then _train_models_callable
    # would load those saved files.
    # For this iteration, since _train_and_evaluate_model_flow still *calls* _load_and_initial_preprocess_data,
    # we'll run the _load_and_initial_preprocess_data as part of the _train_models_callable directly
    # for simplicity to show the new graph without major refactoring of your Python functions.
    # A dedicated _preprocess_data_callable would save and not return directly.
    
    # For now, we will combine preprocessing with training for simpler demonstration.
    # In a full production setup, you would save preprocessed data and preprocessors here.
    
    # 3. Task for model training and evaluation
    train_models = PythonOperator(
        task_id="train_models",
        python_callable=_train_models_callable,
    )

    # Define the task dependencies
    download_data >> train_models