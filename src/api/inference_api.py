import os
import pandas as pd
import numpy as np
import mlflow
import joblib
from flask import Flask, request, jsonify
import traceback # Import for detailed error logging

# --- Configuration ---
# IMPORTANT: Replace with the actual Run ID of your BEST RandomForestClassifier model
# You can find this in your local MLflow UI (e.g., http://localhost:5000)
# Click on the RandomForestClassifier run, and the Run ID will be displayed.
# You provided: "c24f7f7a78684c728bc78b0086894de9" - ensuring this is YOUR RF Run ID is crucial!
MLFLOW_MODEL_RUN_ID = "6c48a932543c4a5eb261fc5eef19ffbf"

# Ensure MLflow is configured to look at your local file store
os.environ['MLFLOW_TRACKING_URI'] = "file://" + os.path.abspath("./mlruns")
print(f"MLflow API loading models from: {os.environ['MLFLOW_TRACKING_URI']}")

# --- Initialize Flask App ---
app = Flask(__name__)

# --- Global variables for loaded model and preprocessors ---
model = None
scaler = None
mlb = None
ohe_user_features = None

# Define the exact list of feature columns in the order expected by the model.
# This list will be dynamically populated after loading preprocessors.
feature_columns_ordered_global = []

def load_model_and_preprocessors():
    """
    Loads the trained model and preprocessing artifacts from MLflow.
    """
    global model, scaler, mlb, ohe_user_features, feature_columns_ordered_global

    print(f"Loading model and preprocessors from MLflow run ID: {MLFLOW_MODEL_RUN_ID}")
    try:
        # Load the model
        # model = mlflow.pyfunc.load_model(f"runs:/{MLFLOW_MODEL_RUN_ID}/model")
        # print("Model loaded successfully.")

        model = mlflow.sklearn.load_model(f"models:/MovieEngagementPredictor/6")
        print("Model loaded successfully.")

        # # Download and load preprocessing artifacts
        # scaler_path = mlflow.artifacts.download_artifacts(f"runs:/{MLFLOW_MODEL_RUN_ID}/preprocessing/scaler.joblib")
        # mlb_path = mlflow.artifacts.download_artifacts(f"runs:/{MLFLOW_MODEL_RUN_ID}/preprocessing/multilabel_binarizer.joblib")
        # ohe_user_path = mlflow.artifacts.download_artifacts(f"runs:/{MLFLOW_MODEL_RUN_ID}/preprocessing/ohe_user.joblib")

        # scaler = joblib.load(scaler_path)
        # mlb = joblib.load(mlb_path)
        # ohe_user_features = joblib.load(ohe_user_path)
        # print("Preprocessing objects loaded successfully.")
        
        # --- CRITICAL: Dynamically determine the exact feature column order and names ---
        # This mirrors the feature_columns list from train.py *after* ReleaseYear has been scaled
        # and its original column name retained for the scaled values.
        
        # These are the column names as they were used *before* scaling in train.py for numerical features.
        # However, for the final model input, these will contain the scaled values.
        # numerical_feature_name = 'ReleaseYear' # The name of the column in the final X, containing scaled values.
        
        # categorical_features_genres = list(mlb.classes_)
        # categorical_features_users = list(ohe_user_features.get_feature_names_out(['Gender', 'Age', 'Occupation']))
        
        # # The order MUST be consistent with how X was formed in train.py
        # feature_columns_ordered_global = [numerical_feature_name] + categorical_features_genres + categorical_features_users
        
        # print(f"Expected feature columns (from loaded preprocessors): {feature_columns_ordered_global}")

    except Exception as e:
        print(f"Error loading model or preprocessors: {e}")
        # In a real application, you might want to raise an exception or exit
        # if the model cannot be loaded, as the API won't function.
        raise RuntimeError(f"Failed to load ML artifacts: {e}")

def preprocess_input(data: dict) -> pd.DataFrame:
    """
    Preprocesses raw input data from the API request into a feature DataFrame.
    This logic must mirror the preprocessing in train.py exactly,
    including column naming conventions for the final DataFrame fed to the model.
    """
    if isinstance(data, dict):
        df_input = pd.DataFrame([data])
    elif isinstance(data, list):
        df_input = pd.DataFrame(data)
    else:
        raise ValueError("Input data must be a dictionary (single instance) or a list of dictionaries (batch).")

    # Ensure all required raw input columns exist in the initial input
    required_raw_cols = ['ReleaseYear', 'Genres', 'Gender', 'Age', 'Occupation']
    if not all(col in df_input.columns for col in required_raw_cols):
        missing_cols = [col for col in required_raw_cols if col not in df_input.columns]
        raise ValueError(f"Input data is missing required columns: {missing_cols}")

    # Make a copy to avoid SettingWithCopyWarning and to build the processed DataFrame
    df_temp_processed = pd.DataFrame(index=df_input.index)

    # 1. Process ReleaseYear: Scale and assign to the 'ReleaseYear' column name
    df_temp_processed['ReleaseYear'] = df_input['ReleaseYear'].astype(float)
    df_temp_processed['ReleaseYear'] = scaler.transform(df_temp_processed[['ReleaseYear']])

    # 2. Process Genres (Multi-label binarization)
    # Ensure 'Genres' column exists and handle potential empty strings
    input_genres = df_input['Genres'].fillna('').astype(str).apply(lambda x: x.split('|') if x else [])
    genre_features = mlb.transform(input_genres)
    genre_feature_df = pd.DataFrame(genre_features, columns=mlb.classes_, index=df_input.index)
    df_temp_processed = pd.concat([df_temp_processed, genre_feature_df], axis=1)

    # 3. Process User Features (One-Hot Encoding for Gender, Age, Occupation)
    # Convert 'Age' to string as OHE was fitted on string/object types
    df_input['Age'] = df_input['Age'].astype(str)
    user_encoded_features = ohe_user_features.transform(df_input[['Gender', 'Age', 'Occupation']])
    user_feature_df = pd.DataFrame(user_encoded_features,
                                   columns=ohe_user_features.get_feature_names_out(['Gender', 'Age', 'Occupation']),
                                   index=df_input.index)
    df_temp_processed = pd.concat([df_temp_processed, user_feature_df], axis=1)

    # Reconstruct the final DataFrame 'X_processed' with the exact columns and order
    # required by the model. This handles cases where some OHE columns might be missing
    # in the current inference input but were present during training.
    X_processed = pd.DataFrame(0.0, index=df_temp_processed.index, columns=feature_columns_ordered_global)
    for col in feature_columns_ordered_global:
        if col in df_temp_processed.columns:
            X_processed[col] = df_temp_processed[col]
    
    return X_processed

# --- Flask Routes ---
@app.route('/')
def health_check():
    """Simple health check endpoint."""
    return "Movie Engagement Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predicts user engagement based on movie and user features.
    Expected JSON input format (for a single prediction):
    {
        "ReleaseYear": 1995,
        "Genres": "Action|Adventure|Fantasy",
        "Gender": "M",
        "Age": 25,
        "Occupation": "student"
    }
    Can also accept a list of such dictionaries for batch prediction.
    """
    if not request.json:
        return jsonify({"error": "Invalid input, please send JSON data"}), 400

    raw_data = request.json
    print(f"Received raw data: {raw_data}")

    try:
        processed_data = preprocess_input(raw_data)
        predictions = model.predict(processed_data)
        labels = ["Dislike", "Like"]
        predicted_labels = [labels[p] for p in predictions]

        return jsonify({"predictions": predicted_labels}), 200

    except ValueError as e:
        # Client-side input validation errors
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        # Any other unexpected internal errors
        traceback.print_exc() # Print full traceback to console/logs for debugging
        return jsonify({"error": f"An internal error occurred: {e}. Check server logs for details."}), 500

# --- Main execution block ---
if __name__ == '__main__':
    # Load model and preprocessors when the app starts
    load_model_and_preprocessors()
    # Run the Flask app
    # Use 0.0.0.0 to make it accessible externally if needed, or 127.0.0.1 for local only
    app.run(host='0.0.0.0', port=5001, debug=True) # debug=True for development, turn off for production