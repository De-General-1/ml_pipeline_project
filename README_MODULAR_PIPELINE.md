# Modular ML Pipeline with Airflow and MLflow

This project implements a modular machine learning pipeline for MovieLens engagement prediction, designed to run on Apache Airflow with MLflow logging and S3 model storage.

## Project Structure

```
ml_pipeline_project/
├── src/
│   └── ml_pipeline/
│       ├── __init__.py
│       ├── config.py              # Centralized configuration
│       ├── data_loader.py         # Data downloading and loading
│       ├── data_preprocessor.py   # Feature engineering and preprocessing
│       └── model_trainer.py       # Model training and evaluation
├── scripts/
│   ├── train.py                   # Original monolithic script
│   ├── train_modular.py          # New modular training script
│   └── test_modular_pipeline.py  # Test script for modular pipeline
├── dags/
│   └── ml_pipeline_dag.py        # Airflow DAG for orchestration
├── models/                        # Local model storage (pickle format)
├── data/                          # Downloaded dataset
├── mlruns/                        # MLflow tracking data
└── requirements.txt               # Dependencies
```

## Key Features

### 1. **Modular Architecture**
- **DataLoader**: Handles downloading and loading MovieLens dataset
- **DataPreprocessor**: Manages feature engineering and preprocessing
- **ModelTrainer**: Handles model training, evaluation, and saving
- **PipelineConfig**: Centralized configuration management

### 2. **Airflow Integration**
The pipeline is orchestrated through Airflow with clear stages:
- **download_data**: Downloads and extracts MovieLens dataset
- **preprocess_data**: Performs feature engineering and preprocessing
- **train_logistic_regression**: Trains Logistic Regression model
- **train_random_forest**: Trains Random Forest model
- **evaluate_models**: Compares and selects best model

### 3. **MLflow Logging**
- Automatic experiment tracking
- Model versioning and registry
- Metrics and parameters logging
- Artifact storage

### 4. **S3 Integration**
- Models saved in pickle format (instead of joblib)
- Automatic upload to S3 with date-based organization
- Configurable bucket and prefix

### 5. **Configuration Management**
All settings centralized in `PipelineConfig`:
- MLflow settings
- Model parameters
- S3 configuration
- Feature engineering settings

## Usage

### 1. Test the Modular Pipeline

```bash
# Test the modular pipeline
python scripts/test_modular_pipeline.py
```

### 2. Run Modular Training

```bash
# Run the complete modular training
python scripts/train_modular.py
```

### 3. Run with Airflow

```bash
# Start Airflow (if using docker-compose)
docker-compose up -d

# The DAG will be automatically picked up from dags/ml_pipeline_dag.py
# Access Airflow UI at http://localhost:8080
```

### 4. Configure S3 (Optional)

Set environment variables for S3 storage:
```bash
export S3_BUCKET="your-s3-bucket-name"
export S3_PREFIX="models"
```

## Pipeline Stages

### Stage 1: Data Loading
- Downloads MovieLens 100K dataset
- Extracts and validates data files
- Returns ratings, movies, and users DataFrames

### Stage 2: Data Preprocessing
- Feature engineering (genres, user features, release year)
- Data binarization (engagement threshold)
- Multi-label encoding for genres
- One-hot encoding for user features
- Standardization of numerical features

### Stage 3: Model Training
- Data splitting (train/test)
- Model training with MLflow logging
- Model evaluation and metrics calculation
- Local saving in pickle format
- S3 upload (if configured)

### Stage 4: Model Evaluation
- Compares model performance
- Selects best model based on F1 score
- Stores evaluation results

## Model Storage

### Local Storage (Pickle Format)
```
models/
├── logisticregression_model.pkl
├── randomforestclassifier_model.pkl
├── scaler.pkl
├── multilabel_binarizer.pkl
└── ohe_user.pkl
```

### S3 Storage (if configured)
```
s3://your-bucket/models/
├── logistic_regression/20240101/
│   ├── model.pkl
│   ├── scaler.pkl
│   ├── multilabel_binarizer.pkl
│   └── ohe_user.pkl
└── random_forest/20240101/
    ├── model.pkl
    ├── scaler.pkl
    ├── multilabel_binarizer.pkl
    └── ohe_user.pkl
```

## Configuration

All settings are managed through `PipelineConfig`:

```python
# MLflow settings
MLFLOW_TRACKING_URI = "file://./mlruns"
MLFLOW_EXPERIMENT_NAME = "Movie-Engagement-Prediction-100K"

# Model parameters
LOGISTIC_REGRESSION_PARAMS = {
    "solver": "liblinear",
    "max_iter": 1000,
    "random_state": 42
}

# S3 settings (from environment)
S3_BUCKET = os.getenv('S3_BUCKET')
S3_PREFIX = os.getenv('S3_PREFIX', 'models')
```

## Benefits of Modular Design

1. **Clear Separation of Concerns**: Each module has a specific responsibility
2. **Reusability**: Components can be used independently
3. **Testability**: Each module can be tested in isolation
4. **Maintainability**: Easy to modify individual components
5. **Scalability**: Easy to add new models or preprocessing steps
6. **Airflow Integration**: Clear task boundaries for orchestration
7. **MLflow Integration**: Proper experiment tracking and model versioning

## Dependencies

Key dependencies added:
- `boto3`: For S3 integration
- `requests`: For data downloading
- `apache-airflow-providers-amazon`: For Airflow S3 integration

## Next Steps

1. **Model Serving**: Add model serving capabilities
2. **Monitoring**: Implement model performance monitoring
3. **A/B Testing**: Add A/B testing framework
4. **Data Validation**: Add data quality checks
5. **CI/CD**: Set up automated testing and deployment 

Did this to get key pair
ssh-keygen -t rsa -b 4096 -f ~/.ssh/ml_pipeline -C "${USER}@ml-pipeline"