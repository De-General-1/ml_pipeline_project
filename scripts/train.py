import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import joblib
import requests
import zipfile
import io

# --- Configuration (kept as global constants for simplicity) ---
MOVIELENS_100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
DOWNLOAD_DIR = "/opt/airflow/data" # Directory inside container
DATA_ROOT_DIR = os.path.join(DOWNLOAD_DIR, "ml-100k") # Expected 'ml-100k' folder from zip extraction

MLFLOW_EXPERIMENT_NAME = "Movie-Engagement-Prediction-100K"
MLFLOW_REGISTERED_MODEL_NAME = "MovieEngagementPredictor"

# Ensure MLflow tracking URI is set for local file store
# Use a path that the Airflow process can definitely write to
mlruns_path = "/opt/airflow/mlruns"
tracking_uri = f"file://{mlruns_path}"

# Ensure the mlruns directory exists
os.makedirs(mlruns_path, exist_ok=True)

# Set the tracking URI in environment and MLflow
os.environ['MLFLOW_TRACKING_URI'] = tracking_uri
mlflow.set_tracking_uri(tracking_uri)

# Verify the tracking URI is set correctly
print(f"MLflow tracking URI set to: {tracking_uri}")
print(f"MLflow tracking URI from mlflow: {mlflow.get_tracking_uri()}")
print(f"MLflow tracking URI from env: {os.environ.get('MLFLOW_TRACKING_URI', 'Not set')}")

def _cleanup_existing_mlflow_data():
    """
    Clean up any existing MLflow data that might cause conflicts.
    This is especially important when running in containers where paths might change.
    """
    try:
        # Delete any existing experiment runs to avoid conflicts
        experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
        if experiment:
            print(f"Found existing experiment: {experiment.experiment_id}")
            # Delete all runs in the experiment
            runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
            if not runs.empty:
                print(f"Cleaning up {len(runs)} existing runs...")
                for run_id in runs['run_id']:
                    try:
                        mlflow.delete_run(run_id)
                    except Exception as e:
                        print(f"Warning: Could not delete run {run_id}: {e}")
    except Exception as e:
        print(f"Warning: Could not cleanup existing MLflow data: {e}")
        # Continue anyway, as this is not critical

def _ensure_mlflow_clean_state():
    """
    Ensure MLflow is in a clean state by setting up tracking URI and cleaning up any conflicts.
    """
    # Force set the tracking URI again to ensure it's correct
    mlruns_path = "/opt/airflow/mlruns"
    tracking_uri = f"file://{mlruns_path}"
    
    # Ensure the directory exists
    os.makedirs(mlruns_path, exist_ok=True)
    
    # Set tracking URI
    os.environ['MLFLOW_TRACKING_URI'] = tracking_uri
    mlflow.set_tracking_uri(tracking_uri)
    
    print(f"Ensured MLflow tracking URI: {tracking_uri}")
    print(f"MLflow tracking URI from mlflow: {mlflow.get_tracking_uri()}")
    
    # Clean up any existing data
    _cleanup_existing_mlflow_data()

def _test_mlflow_setup():
    """
    Test that MLflow can write to the configured location.
    """
    try:
        # Test creating a simple experiment
        test_experiment_name = "test_experiment"
        mlflow.set_experiment(test_experiment_name)
        
        with mlflow.start_run(run_name="test_run") as run:
            mlflow.log_param("test_param", "test_value")
            mlflow.log_metric("test_metric", 1.0)
            print(f"MLflow test successful. Run ID: {run.info.run_id}")
        
        # Clean up test experiment
        experiment = mlflow.get_experiment_by_name(test_experiment_name)
        if experiment:
            runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
            for run_id in runs['run_id']:
                mlflow.delete_run(run_id)
        
        return True
    except Exception as e:
        print(f"MLflow test failed: {e}")
        return False

def _download_and_extract_data(url: str, download_dir: str, data_root_dir: str):
    """
    Downloads a zip file from a URL and extracts its contents.
    Ensures that the final data directory (e.g., ml-100k) is correctly identified.
    """
    os.makedirs(download_dir, exist_ok=True)
    
    zip_file_path = os.path.join(download_dir, os.path.basename(url))

    # Check if the expected data root directory already exists and contains files
    if os.path.exists(data_root_dir) and os.path.isdir(data_root_dir) and \
       os.path.exists(os.path.join(data_root_dir, "u.data")): # Check for a key file
        print(f"Data already extracted to {data_root_dir}. Skipping download and extraction.")
        return data_root_dir

    print(f"Downloading dataset from: {url}")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(zip_file_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Downloaded to: {zip_file_path}")

    print(f"Extracting data to: {download_dir}")
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(download_dir)
    print("Extraction complete.")

    extracted_contents = os.listdir(download_dir)
    
    if os.path.exists(os.path.join(download_dir, "ml-100k", "u.data")):
        final_data_path = os.path.join(download_dir, "ml-100k")
    elif os.path.exists(os.path.join(download_dir, "ml-100k", "ml-100k", "u.data")):
        final_data_path = os.path.join(download_dir, "ml-100k", "ml-100k")
    else:
        raise FileNotFoundError("Could not locate u.data after extraction. "
                                f"Expected it in {download_dir}/ml-100k/u.data or {download_dir}/ml-100k/ml-100k/u.data")

    print(f"Identified base data directory: {final_data_path}")
    return final_data_path

def _load_and_initial_preprocess_data():
    """
    Loads MovieLens 100K dataset from local files and performs initial preprocessing.
    Returns merged DataFrame and the fitted MultiLabelBinarizer and OneHotEncoder.
    Scaler is NOT fitted here as it needs to be fit on train set only.
    """
    base_path = _download_and_extract_data(MOVIELENS_100K_URL, DOWNLOAD_DIR, DATA_ROOT_DIR)
    
    # Load ratings data (u.data)
    ratings_path = os.path.join(base_path, "u.data")
    ratings_df = pd.read_csv(ratings_path, sep='\t', header=None,
                             names=['UserID', 'MovieID', 'Rating', 'Timestamp'])
    print(f"Loaded {len(ratings_df)} ratings entries.")

    # Load movies data (u.item)
    movies_path = os.path.join(base_path, "u.item")
    movies_df = pd.read_csv(movies_path, sep='|', encoding='latin-1', header=None,
                            names=['MovieID', 'Title', 'ReleaseDateStr', 'VideoReleaseDate', 'IMDbURL',
                                   'unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy',
                                   'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
                                   'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'])
    
    genre_cols_100k = ['Action', 'Adventure', 'Animation', "Children's", 'Comedy',
                       'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
                       'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western', 'unknown']
    
    movies_df['Genres'] = movies_df[genre_cols_100k].apply(
        lambda row: '|'.join([col for col, val in row.items() if val == 1]), axis=1
    )
    movies_df.drop(columns=['ReleaseDateStr', 'VideoReleaseDate', 'IMDbURL'] + genre_cols_100k, inplace=True)
    print(f"Loaded {len(movies_df)} movies entries.")

    # Load users data (u.user)
    users_path = os.path.join(base_path, "u.user")
    users_df = pd.read_csv(users_path, sep='|', header=None,
                            names=['UserID', 'Age', 'Gender', 'Occupation', 'Zip-code'])
    print(f"Loaded {len(users_df)} users entries.")

    # Binarize ratings: 'Like' if Rating >= 4, 'Dislike' if Rating < 4
    ratings_df['Engagement'] = (ratings_df['Rating'] >= 4).astype(int)
    print(f"Binarized ratings: {ratings_df['Engagement'].value_counts()}")

    # Merge movies, users, and ratings
    df = pd.merge(ratings_df, movies_df, on='MovieID', how='inner')
    df = pd.merge(df, users_df, on='UserID', how='inner')
    print(f"Merged DataFrame shape: {df.shape}")

    # Feature Engineering:
    # 1. Extract ReleaseYear
    movies_df_for_year = pd.read_csv(movies_path, sep='|', encoding='latin-1', header=None,
                                 names=['MovieID', 'Title', 'ReleaseDateStr', 'VideoReleaseDate', 'IMDbURL'] +
                                         [f'Genre_{i}' for i in range(19)])
    
    movies_df_for_year['ReleaseYear'] = pd.to_datetime(movies_df_for_year['ReleaseDateStr'], format='%d-%b-%Y').dt.year
    
    df = pd.merge(df, movies_df_for_year[['MovieID', 'ReleaseYear']], on='MovieID', how='left')
    df['ReleaseYear'] = df['ReleaseYear'].fillna(df['ReleaseYear'].median())
    df['ReleaseYear'] = df['ReleaseYear'].astype(float)

    # 2. Process Genres (Multi-label binarization)
    mlb = MultiLabelBinarizer()
    # Fit MLB on ALL possible genres from the dataset (or a known list) to ensure consistency
    # using genre_cols_100k ensures all columns are captured.
    mlb.fit([genre_cols_100k]) 
    genre_features = mlb.transform(df['Genres'].apply(lambda x: x.split('|')))
    genre_feature_df = pd.DataFrame(genre_features, columns=mlb.classes_, index=df.index)
    df = pd.concat([df, genre_feature_df], axis=1)

    # 3. Process User Features (One-Hot Encoding for Gender, Age, Occupation)
    ohe_user_features = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    # Fit OHE on the entire set of unique values from users_df to handle all possible categories
    ohe_user_features.fit(users_df[['Gender', 'Age', 'Occupation']]) 

    user_encoded_features = ohe_user_features.transform(df[['Gender', 'Age', 'Occupation']])
    user_feature_df = pd.DataFrame(user_encoded_features,
                                   columns=ohe_user_features.get_feature_names_out(['Gender', 'Age', 'Occupation']),
                                   index=df.index)
    df = pd.concat([df, user_feature_df], axis=1)

    # Define feature columns and target
    numerical_features = ['ReleaseYear']
    categorical_features_genres = list(mlb.classes_)
    categorical_features_users = list(ohe_user_features.get_feature_names_out(['Gender', 'Age', 'Occupation']))

    feature_cols = numerical_features + categorical_features_genres + categorical_features_users
    target_column = 'Engagement'

    df_processed = df[feature_cols + [target_column]].dropna()

    X = df_processed[feature_cols]
    y = df_processed[target_column]

    if X.empty or y.empty:
        raise ValueError("Preprocessed data is empty. Check data loading and cleaning steps.")
    if not np.issubdtype(y.dtype, np.integer):
        raise ValueError("Target variable 'Engagement' is not integer type after binarization.")
    
    print(f"Preprocessed data shape (X, y): {X.shape}, {y.shape}")
    print(f"Features for training: {X.columns.tolist()}") # Log the actual column names

    # Return all necessary components for the next training step
    return X, y, mlb, ohe_user_features

def _train_and_evaluate_model_flow(
    X: pd.DataFrame, 
    y: pd.Series, 
    mlb: MultiLabelBinarizer, 
    ohe_user_features: OneHotEncoder,
    test_size: float = 0.2, 
    random_state: int = 42
):
    """
    Main training and evaluation flow for the ML pipeline.
    This function encapsulates the splitting, scaling, training, and MLflow logging.
    """
    # Ensure MLflow is in a clean state
    _ensure_mlflow_clean_state()
    
    # Test MLflow setup before proceeding
    if not _test_mlflow_setup():
        print("Warning: MLflow setup test failed. Proceeding with training but MLflow logging may fail.")
    
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    logistic_regression_params = {
        "solver": "liblinear",
        "max_iter": 1000,
        "random_state": random_state,
        "n_jobs": 1 # n_jobs > 1 has no effect with liblinear solver
    }

    random_forest_params = {
        "n_estimators": 200,
        "max_depth": 15,
        "min_samples_leaf": 5,
        "random_state": random_state,
        "n_jobs": -1
    }

    print("Splitting data into training and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")

    # Initialize and fit scaler on TRAINING DATA ONLY
    scaler = StandardScaler()
    print("Fitting scaler on 'ReleaseYear' column of training data...")
    scaler.fit(X_train[['ReleaseYear']])

    # Transform ReleaseYear in both train and test sets
    X_train_preprocessed = X_train.copy()
    X_test_preprocessed = X_test.copy()
    
    X_train_preprocessed['ReleaseYear'] = scaler.transform(X_train[['ReleaseYear']])
    X_test_preprocessed['ReleaseYear'] = scaler.transform(X_test[['ReleaseYear']])

    # --- Train and log Logistic Regression model ---
    with mlflow.start_run(run_name="Logistic_Regression_UserFeatures_100K") as run_lr:
        mlflow.log_param("dataset_url", MOVIELENS_100K_URL)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)

        _train_and_log_model(
            X_train_preprocessed, X_test_preprocessed, y_train, y_test,
            scaler, mlb, ohe_user_features, logistic_regression_params,
            model_choice="LogisticRegression"
        )
        print(f"MLflow Run ID (LR): {run_lr.info.run_id}")
        print(f"MLflow UI Link (LR): {mlflow.get_tracking_uri()}/#/experiments/{run_lr.info.experiment_id}/runs/{run_lr.info.run_id}")

    # --- Train and log RandomForestClassifier model ---
    with mlflow.start_run(run_name="Random_Forest_UserFeatures_100K") as run_rf:
        mlflow.log_param("dataset_url", MOVIELENS_100K_URL)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)

        _train_and_log_model(
            X_train_preprocessed, X_test_preprocessed, y_train, y_test,
            scaler, mlb, ohe_user_features, random_forest_params,
            model_choice="RandomForestClassifier"
        )
        print(f"MLflow Run ID (RF): {run_rf.info.run_id}")
        print(f"MLflow UI Link (RF): {mlflow.get_tracking_uri()}/#/experiments/{run_rf.info.experiment_id}/runs/{run_rf.info.run_id}")

    print("Training and evaluation process complete for both models.")


def _train_and_log_model(X_train_processed, X_test_processed, y_train, y_test, scaler, mlb, ohe_user_features, params, model_choice="RandomForestClassifier"):
    """
    Trains a classification model, evaluates it, and logs results to MLflow.
    This is a helper function to be called within an MLflow run.
    """
    if model_choice == "LogisticRegression":
        model = LogisticRegression(**params)
    elif model_choice == "RandomForestClassifier":
        model = RandomForestClassifier(**params)
    else:
        raise ValueError(f"Unknown model_choice: {model_choice}")

    print(f"Training {model_choice} model...")
    model.fit(X_train_processed, y_train)

    print("Evaluating model...")
    y_pred = model.predict(X_test_processed)

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

    mlflow.log_params(params)
    mlflow.log_param("model_choice", model_choice)
    mlflow.log_metrics(metrics)

    # Log preprocessing objects as artifacts within the MLflow run
    # Use a directory under /opt/airflow that the Airflow process can write to
    temp_artifact_dir = "/opt/airflow/tmp_artifacts"
    os.makedirs(temp_artifact_dir, exist_ok=True)
    
    try:
        local_scaler_path = os.path.join(temp_artifact_dir, "scaler.joblib")
        local_mlb_path = os.path.join(temp_artifact_dir, "multilabel_binarizer.joblib")
        local_ohe_user_path = os.path.join(temp_artifact_dir, "ohe_user.joblib")

        joblib.dump(scaler, local_scaler_path)
        joblib.dump(mlb, local_mlb_path)
        joblib.dump(ohe_user_features, local_ohe_user_path)

        mlflow.log_artifact(local_scaler_path, "preprocessing")
        mlflow.log_artifact(local_mlb_path, "preprocessing")
        mlflow.log_artifact(local_ohe_user_path, "preprocessing")

        # Clean up temporary artifacts
        os.remove(local_scaler_path)
        os.remove(local_mlb_path)
        os.remove(local_ohe_user_path)
        
    except Exception as e:
        print(f"Warning: Failed to log preprocessing artifacts: {e}")
        # Continue with model logging even if artifact logging fails
    finally:
        # Clean up temp directory if it's empty
        try:
            if os.path.exists(temp_artifact_dir) and not os.listdir(temp_artifact_dir):
                os.rmdir(temp_artifact_dir)
        except Exception as e:
            print(f"Warning: Failed to clean up temp directory: {e}")

    # Log the model with better error handling
    try:
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=MLFLOW_REGISTERED_MODEL_NAME,
            input_example=X_train_processed[0:1].values.tolist(), # Ensure this is the actual input shape
            signature=mlflow.models.signature.infer_signature(X_train_processed, y_train)
        )
        print("Model successfully logged to MLflow.")
    except Exception as e:
        print(f"Warning: Failed to log model to MLflow: {e}")
        print("Continuing without model logging...")
        # Continue execution even if model logging fails

    