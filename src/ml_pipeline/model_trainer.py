import os
import pickle
import mlflow
import mlflow.sklearn
import boto3
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
from .config import PipelineConfig

class ModelTrainer:
    """Handles model training, evaluation, and saving"""
    
    def __init__(self, experiment_name: str = None, model_name: str = None):
        self.experiment_name = experiment_name or PipelineConfig.MLFLOW_EXPERIMENT_NAME
        self.model_name = model_name or PipelineConfig.MLFLOW_REGISTERED_MODEL_NAME
        self.models_dir = PipelineConfig.MODELS_DIR
        os.makedirs(self.models_dir, exist_ok=True)
        
    def split_data(self, X: pd.DataFrame, y: pd.Series, test_size: float = None, 
                   random_state: int = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        test_size = test_size or PipelineConfig.TEST_SIZE
        random_state = random_state or PipelineConfig.RANDOM_STATE
        """Splits data into training and test sets"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        print(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")
        return X_train, X_test, y_train, y_test
    
    def get_model(self, model_type: str, params: Dict[str, Any] = None):
        """Returns a model instance based on type and parameters"""
        if params is None:
            params = PipelineConfig.get_model_params(model_type)
            
        if model_type == "LogisticRegression":
            return LogisticRegression(**params)
        elif model_type == "RandomForestClassifier":
            return RandomForestClassifier(**params)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
    
    def evaluate_model(self, model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluates model and returns metrics"""
        y_pred = model.predict(X_test)
        
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred, average='weighted'),
            "precision": precision_score(y_test, y_pred, average='weighted'),
            "recall_score": recall_score(y_test, y_pred, average='weighted'),
        }
        
        print(f"Model Metrics: {metrics}")
        return metrics
    
    def log_to_mlflow(self, model, X_train: pd.DataFrame, y_train: pd.Series, 
                      metrics: Dict[str, float], params: Dict[str, Any], 
                      model_type: str, run_name: str):
        """Logs model and metrics to MLflow"""
        mlflow.log_params(params)
        mlflow.log_param("model_type", model_type)
        mlflow.log_metrics(metrics)
        
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=self.model_name,
            input_example=X_train[0:1].values.tolist(),
            signature=mlflow.models.signature.infer_signature(X_train, y_train)
        )
    
    def save_model_locally(self, model, model_type: str, scaler, mlb, ohe_user_features):
        """Saves model and preprocessing artifacts locally in pickle format"""
        # Save model
        model_path = os.path.join(self.models_dir, f"{model_type.lower()}_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Save preprocessing artifacts
        scaler_path = os.path.join(self.models_dir, "scaler.pkl")
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        mlb_path = os.path.join(self.models_dir, "multilabel_binarizer.pkl")
        with open(mlb_path, 'wb') as f:
            pickle.dump(mlb, f)
        
        ohe_path = os.path.join(self.models_dir, "ohe_user.pkl")
        with open(ohe_path, 'wb') as f:
            pickle.dump(ohe_user_features, f)
        
        print(f"Model and artifacts saved locally: {model_path}")
        return model_path, scaler_path, mlb_path, ohe_path
    
    def save_to_s3(self, local_paths: Tuple[str, str, str, str], s3_bucket: str = None, s3_prefix: str = None):
        """Saves model and artifacts to S3"""
        s3_bucket = s3_bucket or PipelineConfig.S3_BUCKET
        if not s3_bucket:
            print("No S3 bucket configured, skipping S3 upload")
            return
            
        session = boto3.Session(profile_name='degen-mlops')
        s3_client = session.client('s3')
        
        model_path, scaler_path, mlb_path, ohe_path = local_paths
        
        # Upload files to S3
        files_to_upload = [
            (model_path, f"{s3_prefix}/model.pkl"),
            (scaler_path, f"{s3_prefix}/scaler.pkl"),
            (mlb_path, f"{s3_prefix}/multilabel_binarizer.pkl"),
            (ohe_path, f"{s3_prefix}/ohe_user.pkl")
        ]
        
        for local_file, s3_key in files_to_upload:
            if os.path.exists(local_file):
                s3_client.upload_file(local_file, s3_bucket, s3_key)
                print(f"Uploaded {local_file} to s3://{s3_bucket}/{s3_key}")
            else:
                print(f"Warning: {local_file} does not exist")
    
    def train_and_evaluate(self, X: pd.DataFrame, y: pd.Series, scaler, mlb, ohe_user_features,
                          model_type: str, params: Dict[str, Any] = None, test_size: float = None,
                          random_state: int = None, s3_bucket: str = None, s3_prefix: str = None) -> Dict[str, Any]:
        """Complete training and evaluation pipeline"""
        
        # Set MLflow experiment
        mlflow.set_experiment(self.experiment_name)
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y, test_size, random_state)
        
        # Transform data
        X_train_transformed = X_train.copy()
        X_train_transformed['ReleaseYear'] = scaler.transform(X_train[['ReleaseYear']])
        X_test_transformed = X_test.copy()
        X_test_transformed['ReleaseYear'] = scaler.transform(X_test[['ReleaseYear']])
        
        # Get and train model
        model = self.get_model(model_type, params)
        print(f"Training {model_type} model...")
        model.fit(X_train_transformed, y_train)
        
        # Evaluate model
        metrics = self.evaluate_model(model, X_test_transformed, y_test)
        
        # Log to MLflow
        run_name = f"{model_type}_{random_state}"
        self.log_to_mlflow(model, X_train_transformed, y_train, metrics, params, model_type, run_name)
        
        # Save locally
        local_paths = self.save_model_locally(model, model_type, scaler, mlb, ohe_user_features)
        
        # Save to S3 if bucket is provided
        if s3_bucket and s3_prefix:
            self.save_to_s3(local_paths, s3_bucket, s3_prefix)
        
        return {
            'model': model,
            'metrics': metrics,
            'local_paths': local_paths,
            'run_name': run_name
        } 