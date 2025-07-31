from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
import os
import sys

# Add src to path
sys.path.append('/opt/airflow/dags/ml_pipeline_project/src')

from ml_pipeline import DataLoader, DataPreprocessor, ModelTrainer
import mlflow
import boto3

# Default arguments
default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# DAG definition
dag = DAG(
    'ml_pipeline_training',
    default_args=default_args,
    description='ML Pipeline for Movie Engagement Prediction',
    schedule_interval='@daily',
    catchup=False,
    tags=['ml', 'training', 'movie-recommendation'],
)

def load_and_preprocess_data(**context):
    """Load and preprocess data"""
    print("Loading and preprocessing data...")
    
    # Initialize components
    data_loader = DataLoader()
    preprocessor = DataPreprocessor()
    
    # Load data
    ratings_df, movies_df, users_df = data_loader.load_data()
    print(f"Loaded data - Ratings: {ratings_df.shape}, Movies: {movies_df.shape}, Users: {users_df.shape}")
    
    # Preprocess data
    X, y, preprocessing_artifacts = preprocessor.preprocess_data(
        ratings_df, movies_df, users_df, data_loader.data_root_dir
    )
    
    # Fit scaler
    scaler = preprocessor.fit_scaler(X)
    
    # Store results in XCom
    return {
        'data_shape': X.shape,
        'target_shape': y.shape,
        'feature_count': len(preprocessing_artifacts['feature_columns'])
    }

def train_logistic_regression(**context):
    """Train Logistic Regression model"""
    print("Training Logistic Regression model...")
    
    # Set MLflow tracking and AWS profile
    os.environ['MLFLOW_TRACKING_URI'] = "file:///opt/airflow/mlruns"
    os.environ['AWS_PROFILE'] = 'degen-mlops'
    
    # Initialize components
    data_loader = DataLoader()
    preprocessor = DataPreprocessor()
    trainer = ModelTrainer()
    
    # Load and preprocess data
    ratings_df, movies_df, users_df = data_loader.load_data()
    X, y, preprocessing_artifacts = preprocessor.preprocess_data(
        ratings_df, movies_df, users_df, data_loader.data_root_dir
    )
    scaler = preprocessor.fit_scaler(X)
    
    # Model parameters
    params = {
        "solver": "liblinear",
        "max_iter": 1000,
        "random_state": 42,
        "n_jobs": -1
    }
    
    # Train model
    with mlflow.start_run(run_name="Airflow_Logistic_Regression") as run:
        mlflow.log_param("dataset_url", data_loader.movielens_url)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("scheduler", "airflow")
        
        result = trainer.train_and_evaluate(
            X, y, scaler, preprocessing_artifacts['mlb'], preprocessing_artifacts['ohe_user_features'],
            "LogisticRegression", params,
            s3_bucket="phase3-mlops-source-bucket-degen-1", 
            s3_prefix="models/logistic_regression"
        )
        
        print(f"Logistic Regression - Accuracy: {result['metrics']['accuracy']:.4f}")
        print(f"MLflow Run ID: {run.info.run_id}")
        
        return {
            'run_id': run.info.run_id,
            'accuracy': result['metrics']['accuracy'],
            'f1_score': result['metrics']['f1_score']
        }

def train_random_forest(**context):
    """Train Random Forest model"""
    print("Training Random Forest model...")
    
    # Set MLflow tracking and AWS profile
    os.environ['MLFLOW_TRACKING_URI'] = "file:///opt/airflow/mlruns"
    os.environ['AWS_PROFILE'] = 'degen-mlops'
    
    # Initialize components
    data_loader = DataLoader()
    preprocessor = DataPreprocessor()
    trainer = ModelTrainer()
    
    # Load and preprocess data
    ratings_df, movies_df, users_df = data_loader.load_data()
    X, y, preprocessing_artifacts = preprocessor.preprocess_data(
        ratings_df, movies_df, users_df, data_loader.data_root_dir
    )
    scaler = preprocessor.fit_scaler(X)
    
    # Model parameters
    params = {
        "n_estimators": 200,
        "max_depth": 15,
        "min_samples_leaf": 5,
        "random_state": 42,
        "n_jobs": -1
    }
    
    # Train model
    with mlflow.start_run(run_name="Airflow_Random_Forest") as run:
        mlflow.log_param("dataset_url", data_loader.movielens_url)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("scheduler", "airflow")
        
        result = trainer.train_and_evaluate(
            X, y, scaler, preprocessing_artifacts['mlb'], preprocessing_artifacts['ohe_user_features'],
            "RandomForestClassifier", params,
            s3_bucket="phase3-mlops-source-bucket-degen-1", 
            s3_prefix="models/random_forest"
        )
        
        print(f"Random Forest - Accuracy: {result['metrics']['accuracy']:.4f}")
        print(f"MLflow Run ID: {run.info.run_id}")
        
        return {
            'run_id': run.info.run_id,
            'accuracy': result['metrics']['accuracy'],
            'f1_score': result['metrics']['f1_score']
        }

def compare_models(**context):
    """Compare model performance and select best model"""
    print("Comparing model performance...")
    
    # Get results from previous tasks
    lr_result = context['task_instance'].xcom_pull(task_ids='train_logistic_regression')
    rf_result = context['task_instance'].xcom_pull(task_ids='train_random_forest')
    
    print(f"Logistic Regression - Accuracy: {lr_result['accuracy']:.4f}, F1: {lr_result['f1_score']:.4f}")
    print(f"Random Forest - Accuracy: {rf_result['accuracy']:.4f}, F1: {rf_result['f1_score']:.4f}")
    
    # Select best model based on F1 score
    if lr_result['f1_score'] > rf_result['f1_score']:
        best_model = 'LogisticRegression'
        best_run_id = lr_result['run_id']
        best_score = lr_result['f1_score']
    else:
        best_model = 'RandomForestClassifier'
        best_run_id = rf_result['run_id']
        best_score = rf_result['f1_score']
    
    print(f"Best model: {best_model} with F1 score: {best_score:.4f}")
    print(f"Best model run ID: {best_run_id}")
    
    return {
        'best_model': best_model,
        'best_run_id': best_run_id,
        'best_f1_score': best_score
    }

# Define tasks
data_preprocessing_task = PythonOperator(
    task_id='load_and_preprocess_data',
    python_callable=load_and_preprocess_data,
    dag=dag,
)

train_lr_task = PythonOperator(
    task_id='train_logistic_regression',
    python_callable=train_logistic_regression,
    dag=dag,
)

train_rf_task = PythonOperator(
    task_id='train_random_forest',
    python_callable=train_random_forest,
    dag=dag,
)

model_comparison_task = PythonOperator(
    task_id='compare_models',
    python_callable=compare_models,
    dag=dag,
)

# Set task dependencies
data_preprocessing_task >> [train_lr_task, train_rf_task] >> model_comparison_task