import os
import sys
import mlflow

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from ml_pipeline import DataLoader, DataPreprocessor, ModelTrainer

# MLflow Setup
os.environ['MLFLOW_TRACKING_URI'] = "file://" + os.path.abspath("./mlruns")
os.environ['AWS_PROFILE'] = 'degen-mlops'
mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
print(f"MLflow tracking URI: {os.environ['MLFLOW_TRACKING_URI']}")
print(f"MLflow artifacts will be stored in S3: s3://phase3-mlops-source-bucket-degen-1/mlflow-artifacts")

def main():
    """Main training pipeline"""
    
    # Initialize components
    data_loader = DataLoader()
    preprocessor = DataPreprocessor()
    trainer = ModelTrainer()
    
    # Model parameters
    logistic_regression_params = {
        "solver": "liblinear",
        "max_iter": 1000,
        "random_state": 42,
        "n_jobs": -1
    }

    random_forest_params = {
        "n_estimators": 200,
        "max_depth": 15,
        "min_samples_leaf": 5,
        "random_state": 42,
        "n_jobs": -1
    }
    
    # Load data
    print("Loading data...")
    ratings_df, movies_df, users_df = data_loader.load_data()
    
    # Preprocess data
    print("Preprocessing data...")
    X, y, preprocessing_artifacts = preprocessor.preprocess_data(
        ratings_df, movies_df, users_df, data_loader.data_root_dir
    )
    
    # Fit scaler
    print("Fitting scaler...")
    scaler = preprocessor.fit_scaler(X)
    
    # Train Logistic Regression
    print("Training Logistic Regression...")
    with mlflow.start_run(run_name="Logistic_Regression_Modular") as run_lr:
        mlflow.log_param("dataset_url", data_loader.movielens_url)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)
        
        result_lr = trainer.train_and_evaluate(
            X, y, scaler, preprocessing_artifacts['mlb'], preprocessing_artifacts['ohe_user_features'],
            "LogisticRegression", logistic_regression_params,
            s3_bucket="phase3-mlops-source-bucket-degen-1", s3_prefix="models/logistic_regression"
        )
        
        print(f"MLflow Run ID (LR): {run_lr.info.run_id}")
    
    # Train Random Forest
    print("Training Random Forest...")
    with mlflow.start_run(run_name="Random_Forest_Modular") as run_rf:
        mlflow.log_param("dataset_url", data_loader.movielens_url)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)
        
        result_rf = trainer.train_and_evaluate(
            X, y, scaler, preprocessing_artifacts['mlb'], preprocessing_artifacts['ohe_user_features'],
            "RandomForestClassifier", random_forest_params,
            s3_bucket="phase3-mlops-source-bucket-degen-1", s3_prefix="models/random_forest"
        )
        
        print(f"MLflow Run ID (RF): {run_rf.info.run_id}")
    
    print("Training pipeline completed successfully!")

if __name__ == "__main__":
    main() 