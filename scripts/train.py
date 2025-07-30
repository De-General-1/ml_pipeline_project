import os
import pandas as pd
import numpy as np
import kagglehub
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import joblib
import dagshub

# DagsHub MLflow setup
# Ensure you have set your MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_PASSWORD
# (DagsHub API token) as environment variables, or hardcode them for local testing
# if you understand the security implications.
# For local development, if you run 'dagshub login' it often handles credentials.
dagshub.init(repo_owner='De-General-1', repo_name='ml_pipeline_project', mlflow=True)
os.environ['MLFLOW_TRACKING_URI'] = f"https://dagshub.com/De-General-1/mlops-music.mlflow"
# Optional: Set username/password env vars directly in your script for testing
# os.environ['MLFLOW_TRACKING_USERNAME'] = 'De-General-1'
# os.environ['MLFLOW_TRACKING_PASSWORD'] = 'YOUR_DAGSHUB_API_TOKEN' # ONLY FOR LOCAL TESTING, USE ENV VARS IN PROD

MLFLOW_EXPERIMENT_NAME = "Movie-Engagement-Prediction"
MLFLOW_REGISTERED_MODEL_NAME = "MovieEngagementPredictor"
LOCAL_MODEL_PATH = "./models/logistic_regression_model.joblib"
LOCAL_SCALER_PATH = "./models/scaler.joblib"
LOCAL_MLB_PATH = "./models/multilabel_binarizer.joblib"


def load_and_preprocess_data(dataset_name_param: str):
    """
    Downloads MovieLens 1M dataset, loads movies and ratings,
    and preprocesses them for training.
    """
    print(f"Downloading dataset from Kaggle: {dataset_name_param}")
    try:
        path = kagglehub.dataset_download(dataset_name_param)
        print(f"Dataset downloaded to: {path}")

        # Load movies data
        movies_csv_path = os.path.join(path, "movies.dat")
        # MovieLens .dat files are pipe-separated with no header and specific encoding
        movies_df = pd.read_csv(movies_csv_path, sep='::', engine='python', 
                                header=None, names=['MovieID', 'Title', 'Genres'], encoding='latin-1')
        print(f"Loaded {len(movies_df)} movies entries.")

        # Load ratings data
        ratings_csv_path = os.path.join(path, "ratings.dat")
        ratings_df = pd.read_csv(ratings_csv_path, sep='::', engine='python',
                                 header=None, names=['UserID', 'MovieID', 'Rating', 'Timestamp'], encoding='latin-1')
        print(f"Loaded {len(ratings_df)} ratings entries.")

        # Binarize ratings: 'Like' if Rating >= 4, 'Dislike' if Rating < 4
        ratings_df['Engagement'] = (ratings_df['Rating'] >= 4).astype(int)
        print(f"Binarized ratings: {ratings_df['Engagement'].value_counts()}")

        # Merge movies and ratings
        df = pd.merge(ratings_df, movies_df, on='MovieID', how='inner')
        print(f"Merged DataFrame shape: {df.shape}")

        # Feature Engineering:
        # 1. Extract release year (simple for now, could parse full title)
        df['ReleaseYear'] = df['Title'].str.extract(r'\((\d{4})\)').astype(float)
        # Handle cases where year might not be found or is problematic
        df['ReleaseYear'].fillna(df['ReleaseYear'].median(), inplace=True)
        
        # 2. Process Genres (Multi-label binarization)
        mlb = MultiLabelBinarizer()
        df_genres = df['Genres'].str.split('|').apply(lambda x: mlb.fit_transform([x])[0])
        # To make it work correctly with Series.apply, we need to create a new DataFrame
        # and then concatenate it.
        # First, fit on all unique genres
        all_genres = set('|'.join(df['Genres'].unique()).split('|'))
        mlb.fit([list(all_genres)]) # Fit MLB on all possible genres

        genre_features = mlb.transform(df['Genres'].apply(lambda x: x.split('|')))
        genre_feature_df = pd.DataFrame(genre_features, columns=mlb.classes_, index=df.index)
        
        df = pd.concat([df, genre_feature_df], axis=1)

        # Select features and target
        # Features: ReleaseYear, and all binarized genre columns
        feature_columns = ['ReleaseYear'] + list(mlb.classes_)
        target_column = 'Engagement'

        # Drop rows with any NaN in selected features (should be handled by fillna for ReleaseYear)
        df.dropna(subset=feature_columns + [target_column], inplace=True)
        
        X = df[feature_columns]
        y = df[target_column]

        print(f"Features for training: {feature_columns}")
        print(f"Preprocessed data shape (X, y): {X.shape}, {y.shape}")

        # Data Validation (simple checks)
        if X.empty or y.empty:
            raise ValueError("Preprocessed data is empty. Check data loading and cleaning steps.")
        if not np.issubdtype(y.dtype, np.integer):
            raise ValueError("Target variable 'Engagement' is not integer type after binarization.")

        return X, y, mlb

    except Exception as e:
        print(f"Error loading or preprocessing dataset: {e}")
        raise


def train_and_evaluate_model(X_train, X_test, y_train, y_test, scaler, mlb, params):
    """
    Trains a Logistic Regression model, evaluates it, and logs results to MLflow.
    """
    print("Scaling features...")
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LogisticRegression(**params)
    print("Training model...")
    model.fit(X_train_scaled, y_train)

    print("Evaluating model...")
    y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')

    metrics = {
        "accuracy": accuracy,
        "f1_score": f1,
        "precision": precision,
        "recall_score": recall,
    }

    print(f"Model Metrics: {metrics}")

    # Log metrics and parameters to MLflow
    mlflow.log_params(params)
    mlflow.log_metrics(metrics)

    # Log the scikit-learn model
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name=MLFLOW_REGISTERED_MODEL_NAME,
        input_example=X_train_scaled[0:1].tolist(), # Log a sample input for inference
        signature=mlflow.models.signature.infer_signature(X_train_scaled, y_train)
    )

    # Save the scaler and MLB as artifacts for later use in inference
    joblib.dump(scaler, LOCAL_SCALER_PATH)
    joblib.dump(mlb, LOCAL_MLB_PATH)
    mlflow.log_artifact(LOCAL_SCALER_PATH, "preprocessing")
    mlflow.log_artifact(LOCAL_MLB_PATH, "preprocessing")
    
    print("Model, scaler, and MLB logged to MLflow and saved locally.")
    return model, metrics


if __name__ == "__main__":
    dataset_name = "odedgolden/movielens-1m-dataset"
    test_size = 0.2
    random_state = 42

    # Model hyperparameters for Logistic Regression
    model_params = {
        "solver": "liblinear", # Good for small datasets, supports L1/L2
        "max_iter": 1000,
        "random_state": random_state
    }

    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        mlflow.log_param("dataset_name", dataset_name)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)

        # Load and preprocess data
        X, y, mlb = load_and_preprocess_data(dataset_name)

        # Split data
        print("Splitting data into training and test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        print(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")

        # Initialize and fit scaler on training data
        scaler = StandardScaler()
        print("Fitting scaler on training data...")
        scaler.fit(X_train[['ReleaseYear']]) # Fit only on numerical columns first if needed, then transform all
        
        # We need to apply scaling to the numeric column 'ReleaseYear'
        # The genre columns are already 0/1, so no scaling needed for them.
        X_train_preprocessed = X_train.copy()
        X_test_preprocessed = X_test.copy()
        
        X_train_preprocessed['ReleaseYear'] = scaler.transform(X_train[['ReleaseYear']])
        X_test_preprocessed['ReleaseYear'] = scaler.transform(X_test[['ReleaseYear']])
        
        # Train and evaluate
        model, metrics = train_and_evaluate_model(
            X_train_preprocessed, X_test_preprocessed, y_train, y_test, scaler, mlb, model_params
        )

        print(f"MLflow Run ID: {run.info.run_id}")
        print(f"MLflow UI Link: {mlflow.get_tracking_uri()}/#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}")
        
        # Save the model locally as a joblib file
        # Ensure the 'models' directory exists
        os.makedirs("./models", exist_ok=True)
        joblib.dump(model, LOCAL_MODEL_PATH)
        print(f"Model saved locally to {LOCAL_MODEL_PATH}")

    print("Training and evaluation process complete.")