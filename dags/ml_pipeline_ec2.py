from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import os
import sys
import boto3
import mlflow
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pickle
import requests
import zipfile
from io import BytesIO

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
    'ml_pipeline_ec2_training',
    default_args=default_args,
    description='ML Pipeline for Movie Engagement Prediction on EC2',
    schedule_interval='@daily',
    catchup=False,
    tags=['ml', 'training', 'movie-recommendation', 'ec2'],
)

def download_and_extract_data(**context):
    """Download and extract MovieLens data"""
    print("Downloading MovieLens 100K dataset...")
    
    url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
    data_dir = "/tmp/ml-100k"
    
    # Download and extract
    response = requests.get(url)
    with zipfile.ZipFile(BytesIO(response.content)) as zip_file:
        zip_file.extractall("/tmp")
    
    print(f"Data extracted to {data_dir}")
    return data_dir

def load_and_preprocess_data(**context):
    """Load and preprocess data"""
    print("Loading and preprocessing data...")
    
    data_dir = "/tmp/ml-100k"
    
    # Load ratings
    ratings_df = pd.read_csv(f"{data_dir}/u.data", sep='\t', 
                           names=['UserID', 'MovieID', 'Rating', 'Timestamp'])
    
    # Load movies
    movies_df = pd.read_csv(f"{data_dir}/u.item", sep='|', encoding='latin-1', header=None,
                          names=['MovieID', 'Title', 'ReleaseDateStr', 'VideoReleaseDate', 'IMDbURL'] +
                                [f'Genre_{i}' for i in range(19)])
    
    # Load users
    users_df = pd.read_csv(f"{data_dir}/u.user", sep='|',
                         names=['UserID', 'Age', 'Gender', 'Occupation', 'ZipCode'])
    
    # Process genres
    genre_cols = ['Action', 'Adventure', 'Animation', "Children's", 'Comedy',
                  'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
                  'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western', 'unknown']
    
    movies_df.columns = ['MovieID', 'Title', 'ReleaseDateStr', 'VideoReleaseDate', 'IMDbURL'] + genre_cols
    movies_df['Genres'] = movies_df[genre_cols].apply(
        lambda row: '|'.join([col for col, val in row.items() if val == 1]), axis=1
    )
    
    # Extract release year
    movies_df['ReleaseYear'] = pd.to_datetime(movies_df['ReleaseDateStr'], format='%d-%b-%Y', errors='coerce').dt.year
    movies_df['ReleaseYear'] = movies_df['ReleaseYear'].fillna(movies_df['ReleaseYear'].median())
    
    # Binarize ratings (engagement)
    ratings_df['Engagement'] = (ratings_df['Rating'] >= 4).astype(int)
    
    # Merge datasets
    df = pd.merge(ratings_df, movies_df[['MovieID', 'ReleaseYear', 'Genres']], on='MovieID', how='inner')
    df = pd.merge(df, users_df, on='UserID', how='inner')
    
    print(f"Merged data shape: {df.shape}")
    print(f"Engagement distribution: {df['Engagement'].value_counts()}")
    
    return {
        'data_shape': df.shape,
        'engagement_dist': df['Engagement'].value_counts().to_dict()
    }

def train_models(**context):
    """Train both models and select the best one"""
    print("Training models...")
    
    # Set MLflow tracking
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("Movie-Engagement-Prediction-EC2")
    
    data_dir = "/tmp/ml-100k"
    s3_bucket = os.getenv('S3_BUCKET', 'phase3-mlops-source-bucket-degen-1')
    
    # Load and preprocess data (simplified version)
    ratings_df = pd.read_csv(f"{data_dir}/u.data", sep='\t', 
                           names=['UserID', 'MovieID', 'Rating', 'Timestamp'])
    movies_df = pd.read_csv(f"{data_dir}/u.item", sep='|', encoding='latin-1', header=None,
                          names=['MovieID', 'Title', 'ReleaseDateStr', 'VideoReleaseDate', 'IMDbURL'] +
                                [f'Genre_{i}' for i in range(19)])
    users_df = pd.read_csv(f"{data_dir}/u.user", sep='|',
                         names=['UserID', 'Age', 'Gender', 'Occupation', 'ZipCode'])
    
    # Quick preprocessing
    genre_cols = ['Action', 'Adventure', 'Animation', "Children's", 'Comedy',
                  'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
                  'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western', 'unknown']
    
    movies_df.columns = ['MovieID', 'Title', 'ReleaseDateStr', 'VideoReleaseDate', 'IMDbURL'] + genre_cols
    movies_df['ReleaseYear'] = pd.to_datetime(movies_df['ReleaseDateStr'], format='%d-%b-%Y', errors='coerce').dt.year
    movies_df['ReleaseYear'] = movies_df['ReleaseYear'].fillna(movies_df['ReleaseYear'].median())
    
    ratings_df['Engagement'] = (ratings_df['Rating'] >= 4).astype(int)
    
    # Merge and create features
    df = pd.merge(ratings_df, movies_df, on='MovieID', how='inner')
    df = pd.merge(df, users_df, on='UserID', how='inner')
    
    # Create feature matrix
    mlb = MultiLabelBinarizer()
    mlb.fit([genre_cols])
    
    ohe_user = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    ohe_user.fit(users_df[['Gender', 'Age', 'Occupation']])
    
    # Process genres
    genre_features = mlb.transform(df[genre_cols].apply(lambda row: [col for col, val in row.items() if val == 1], axis=1))
    genre_df = pd.DataFrame(genre_features, columns=mlb.classes_, index=df.index)
    
    # Process user features
    user_features = ohe_user.transform(df[['Gender', 'Age', 'Occupation']])
    user_df = pd.DataFrame(user_features, columns=ohe_user.get_feature_names_out(['Gender', 'Age', 'Occupation']), index=df.index)
    
    # Combine features
    feature_df = pd.concat([df[['ReleaseYear']], genre_df, user_df], axis=1)
    X = feature_df
    y = df['Engagement']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = X.copy()
    X_scaled['ReleaseYear'] = scaler.fit_transform(X[['ReleaseYear']])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    
    models = {
        'LogisticRegression': LogisticRegression(solver='liblinear', max_iter=1000, random_state=42),
        'RandomForestClassifier': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    }
    
    best_model = None
    best_score = 0
    best_model_name = None
    
    # Train models
    for model_name, model in models.items():
        with mlflow.start_run(run_name=f"EC2_{model_name}") as run:
            print(f"Training {model_name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Log metrics
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("test_size", 0.2)
            mlflow.log_param("random_state", 42)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("f1_score", f1)
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            print(f"{model_name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
            
            # Track best model
            if f1 > best_score:
                best_score = f1
                best_model = model
                best_model_name = model_name
    
    # Save best model to S3
    if best_model:
        print(f"Best model: {best_model_name} with F1 score: {best_score:.4f}")
        
        s3_client = boto3.client('s3')
        
        # Save model artifacts
        artifacts = {
            'model.pkl': best_model,
            'scaler.pkl': scaler,
            'multilabel_binarizer.pkl': mlb,
            'ohe_user.pkl': ohe_user
        }
        
        for artifact_name, artifact in artifacts.items():
            # Save locally first
            local_path = f"/tmp/{artifact_name}"
            with open(local_path, 'wb') as f:
                pickle.dump(artifact, f)
            
            # Upload to S3
            s3_key = f"models/{best_model_name.lower()}/{artifact_name}"
            s3_client.upload_file(local_path, s3_bucket, s3_key)
            print(f"Uploaded {artifact_name} to s3://{s3_bucket}/{s3_key}")
    
    return {
        'best_model': best_model_name,
        'best_score': best_score,
        'models_trained': list(models.keys())
    }

# Define tasks
download_task = PythonOperator(
    task_id='download_data',
    python_callable=download_and_extract_data,
    dag=dag,
)

preprocess_task = PythonOperator(
    task_id='preprocess_data',
    python_callable=load_and_preprocess_data,
    dag=dag,
)

train_task = PythonOperator(
    task_id='train_models',
    python_callable=train_models,
    dag=dag,
)

# Set dependencies
download_task >> preprocess_task >> train_task