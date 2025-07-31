import os
from typing import Dict, Any

class PipelineConfig:
    """Configuration class for ML pipeline settings"""
    
    # MLflow settings
    MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', "file://./mlruns")
    MLFLOW_EXPERIMENT_NAME = "Movie-Engagement-Prediction-100K"
    MLFLOW_REGISTERED_MODEL_NAME = "MovieEngagementPredictor"
    
    # Data settings
    MOVIELENS_100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
    DOWNLOAD_DIR = "./data"
    DATA_ROOT_DIR = os.path.join(DOWNLOAD_DIR, "ml-100k")
    
    # Model settings
    MODELS_DIR = "./models"
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    
    # S3 settings
    S3_BUCKET = os.getenv('S3_BUCKET', 'phase3-mlops-source-bucket-degen-1')
    S3_PREFIX = os.getenv('S3_PREFIX', 'models')
    S3_DATA_PREFIX = 'data'
    
    # Model parameters
    LOGISTIC_REGRESSION_PARAMS = {
        "solver": "liblinear",
        "max_iter": 1000,
        "random_state": RANDOM_STATE,
        "n_jobs": -1
    }
    
    RANDOM_FOREST_PARAMS = {
        "n_estimators": 200,
        "max_depth": 15,
        "min_samples_leaf": 5,
        "random_state": RANDOM_STATE,
        "n_jobs": -1
    }
    
    # Feature engineering settings
    GENRE_COLS_100K = [
        'Action', 'Adventure', 'Animation', "Children's", 'Comedy',
        'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
        'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western', 'unknown'
    ]
    
    # Engagement threshold
    ENGAGEMENT_THRESHOLD = 4  # Rating >= 4 is considered engagement
    
    @classmethod
    def get_model_params(cls, model_type: str) -> Dict[str, Any]:
        """Get model parameters by type"""
        if model_type == "LogisticRegression":
            return cls.LOGISTIC_REGRESSION_PARAMS
        elif model_type == "RandomForestClassifier":
            return cls.RANDOM_FOREST_PARAMS
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
    
    @classmethod
    def get_s3_path(cls, model_type: str, date_str: str = None) -> str:
        """Get S3 path for model storage"""
        if date_str is None:
            from datetime import datetime
            date_str = datetime.now().strftime('%Y%m%d')
        
        return f"{cls.S3_PREFIX}/{model_type.lower()}/{date_str}" 