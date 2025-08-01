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

# --- MLflow Setup ---
os.environ['MLFLOW_TRACKING_URI'] = "file://" + os.path.abspath("./mlruns")
print(f"MLflow tracking to local file store: {os.environ['MLFLOW_TRACKING_URI']}")

MLFLOW_EXPERIMENT_NAME = "Movie-Engagement-Prediction-100K"
MLFLOW_REGISTERED_MODEL_NAME = "MovieEngagementPredictor"
LOCAL_MODEL_PATH = "./models/model.joblib"
LOCAL_SCALER_PATH = "./models/scaler.joblib"
LOCAL_MLB_PATH = "./models/multilabel_binarizer.joblib"
LOCAL_OHE_USER_PATH = "./models/ohe_user.joblib"

MOVIELENS_100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
DOWNLOAD_DIR = "./data" # Directory to save the downloaded zip
# Define the expected *final* directory where u.data, u.item, etc., will reside
DATA_ROOT_DIR = os.path.join(DOWNLOAD_DIR, "ml-100k") # Expected 'ml-100k' folder from zip extraction

def download_and_extract_data(url: str, download_dir: str, data_root_dir: str):
    """
    Downloads a zip file from a URL and extracts its contents.
    Ensures that the final data directory (e.g., ml-100k) is correctly identified.
    """
    os.makedirs(download_dir, exist_ok=True)
    
    zip_file_path = os.path.join(download_dir, os.path.basename(url))

    # Check if the expected data root directory already exists and contains files
    # This prevents re-downloading/extracting if already done
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

    print(f"Extracting data to: {download_dir}") # Extract into DOWNLOAD_DIR
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(download_dir)
    print("Extraction complete.")

    # After extraction, the actual data files (u.data, u.item etc.) might be
    # directly in 'download_dir/ml-100k/' or 'download_dir/ml-100k/ml-100k/'
    # Let's find the correct base path dynamically
    extracted_contents = os.listdir(download_dir)
    
    # Common scenario: ml-100k.zip extracts directly into a 'ml-100k' folder
    # which contains u.data etc.
    if os.path.exists(os.path.join(download_dir, "ml-100k", "u.data")):
        final_data_path = os.path.join(download_dir, "ml-100k")
    # Less common but possible: ml-100k.zip extracts 'ml-100k/ml-100k/u.data'
    elif os.path.exists(os.path.join(download_dir, "ml-100k", "ml-100k", "u.data")):
        final_data_path = os.path.join(download_dir, "ml-100k", "ml-100k")
    else:
        raise FileNotFoundError("Could not locate u.data after extraction. "
                                f"Expected it in {download_dir}/ml-100k/u.data or {download_dir}/ml-100k/ml-100k/u.data")

    print(f"Identified base data directory: {final_data_path}")
    return final_data_path


def load_and_preprocess_data():
    """
    Loads MovieLens 100K dataset from local files and preprocesses them for training.
    """
    base_path = download_and_extract_data(MOVIELENS_100K_URL, DOWNLOAD_DIR, DATA_ROOT_DIR)
    
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
    # 1. Extract ReleaseYear (using the ReleaseDateStr for this dataset)
    # Re-read u.item to get ReleaseDateStr without dropping it early for year extraction.
    movies_df_for_year = pd.read_csv(movies_path, sep='|', encoding='latin-1', header=None,
                                names=['MovieID', 'Title', 'ReleaseDateStr', 'VideoReleaseDate', 'IMDbURL'] +
                                       [f'Genre_{i}' for i in range(19)]) # Placeholder names
    
    movies_df_for_year['ReleaseYear'] = pd.to_datetime(movies_df_for_year['ReleaseDateStr'], format='%d-%b-%Y').dt.year
    
    # Merge this ReleaseYear from movies_df_for_year into df
    df = pd.merge(df, movies_df_for_year[['MovieID', 'ReleaseYear']], on='MovieID', how='left')
    df['ReleaseYear'] = df['ReleaseYear'].fillna(df['ReleaseYear'].median())
    df['ReleaseYear'] = df['ReleaseYear'].astype(float)

    # 2. Process Genres (Multi-label binarization)
    mlb = MultiLabelBinarizer()
    mlb.fit([genre_cols_100k])
    genre_features = mlb.transform(df['Genres'].apply(lambda x: x.split('|')))
    genre_feature_df = pd.DataFrame(genre_features, columns=mlb.classes_, index=df.index)
    df = pd.concat([df, genre_feature_df], axis=1)

    # 3. Process User Features (One-Hot Encoding for Gender, Age, Occupation)
    ohe_user_features = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    ohe_user_features.fit(users_df[['Gender', 'Age', 'Occupation']])

    user_encoded_features = ohe_user_features.transform(df[['Gender', 'Age', 'Occupation']])
    user_feature_df = pd.DataFrame(user_encoded_features,
                                   columns=ohe_user_features.get_feature_names_out(['Gender', 'Age', 'Occupation']),
                                   index=df.index)
    df = pd.concat([df, user_feature_df], axis=1)

    # Select features and target
    numerical_features = ['ReleaseYear']
    categorical_features_genres = list(mlb.classes_)
    categorical_features_users = list(ohe_user_features.get_feature_names_out(['Gender', 'Age', 'Occupation']))

    feature_columns = numerical_features + categorical_features_genres + categorical_features_users
    target_column = 'Engagement'

    df_processed = df[feature_columns + [target_column]].dropna()

    X = df_processed[feature_columns]
    y = df_processed[target_column]

    print(f"Features for training: {feature_columns}")
    print(f"Preprocessed data shape (X, y): {X.shape}, {y.shape}")

    if X.empty or y.empty:
        raise ValueError("Preprocessed data is empty. Check data loading and cleaning steps.")
    if not np.issubdtype(y.dtype, np.integer):
        raise ValueError("Target variable 'Engagement' is not integer type after binarization.")

    return X, y, mlb, ohe_user_features


def train_and_evaluate_model(X_train_processed, X_test_processed, y_train, y_test, scaler, mlb, ohe_user_features, params, model_choice="RandomForestClassifier"):
    """
    Trains a classification model, evaluates it, and logs results to MLflow.
    X_train_processed and X_test_processed are expected to have 'ReleaseYear' scaled.
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

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name=MLFLOW_REGISTERED_MODEL_NAME,
        input_example=X_train_processed[0:1].values.tolist(),
        signature=mlflow.models.signature.infer_signature(X_train_processed, y_train)
    )

    os.makedirs("./models", exist_ok=True)
    
    joblib.dump(scaler, LOCAL_SCALER_PATH)
    joblib.dump(mlb, LOCAL_MLB_PATH)
    joblib.dump(ohe_user_features, LOCAL_OHE_USER_PATH)
    mlflow.log_artifact(LOCAL_SCALER_PATH, "preprocessing")
    mlflow.log_artifact(LOCAL_MLB_PATH, "preprocessing")
    mlflow.log_artifact(LOCAL_OHE_USER_PATH, "preprocessing")

    print("Model, scaler, MLB, and OHE logged to MLflow artifacts (and registry locally).")
    return model, metrics


if __name__ == "__main__":
    test_size = 0.2
    random_state = 42

    logistic_regression_params = {
        "solver": "liblinear",
        "max_iter": 1000,
        "random_state": random_state,
        "n_jobs": -1
    }

    random_forest_params = {
        "n_estimators": 200,
        "max_depth": 15,
        "min_samples_leaf": 5,
        "random_state": random_state,
        "n_jobs": -1
    }

    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # --- Run for Logistic Regression (Optional - can comment out for faster testing) ---
    with mlflow.start_run(run_name="Logistic_Regression_UserFeatures_100K") as run_lr:
        mlflow.log_param("dataset_url", MOVIELENS_100K_URL)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)

        X, y, mlb, ohe_user_features = load_and_preprocess_data()

        print("Splitting data into training and test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        print(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")

        scaler = StandardScaler()
        print("Fitting scaler on 'ReleaseYear' column of training data...")
        scaler.fit(X_train[['ReleaseYear']])

        X_train_preprocessed = X_train.copy()
        X_test_preprocessed = X_test.copy()

        X_train_preprocessed['ReleaseYear'] = scaler.transform(X_train[['ReleaseYear']])
        X_test_preprocessed['ReleaseYear'] = scaler.transform(X_test[['ReleaseYear']])

        model_lr, metrics_lr = train_and_evaluate_model(
            X_train_preprocessed, X_test_preprocessed, y_train, y_test,
            scaler, mlb, ohe_user_features, logistic_regression_params,
            model_choice="LogisticRegression"
        )

        print(f"MLflow Run ID (LR): {run_lr.info.run_id}")
        print(f"MLflow UI Link (LR): {mlflow.get_tracking_uri()}/#/experiments/{run_lr.info.experiment_id}/runs/{run_lr.info.run_id}")

        joblib.dump(model_lr, LOCAL_MODEL_PATH.replace(".joblib", "_lr.joblib"))

    # --- Run for RandomForestClassifier (Primary focus) ---
    with mlflow.start_run(run_name="Random_Forest_UserFeatures_100K") as run_rf:
        mlflow.log_param("dataset_url", MOVIELENS_100K_URL)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)

        model_rf, metrics_rf = train_and_evaluate_model(
            X_train_preprocessed, X_test_preprocessed, y_train, y_test,
            scaler, mlb, ohe_user_features, random_forest_params,
            model_choice="RandomForestClassifier"
        )

        print(f"MLflow Run ID (RF): {run_rf.info.run_id}")
        print(f"MLflow UI Link (RF): {mlflow.get_tracking_uri()}/#/experiments/{run_rf.info.experiment_id}/runs/{run_rf.info.run_id}")

        joblib.dump(model_rf, LOCAL_MODEL_PATH.replace(".joblib", "_rf.joblib"))

    print("Training and evaluation process complete for both models.")