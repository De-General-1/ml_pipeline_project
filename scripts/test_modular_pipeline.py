#!/usr/bin/env python3
"""
Test script for the modular ML pipeline
"""

import os
import sys
import mlflow

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from ml_pipeline import DataLoader, DataPreprocessor, ModelTrainer, PipelineConfig

def test_pipeline():
    """Test the complete pipeline"""
    
    print("Testing modular ML pipeline...")
    
    # Set MLflow tracking
    os.environ['MLFLOW_TRACKING_URI'] = PipelineConfig.MLFLOW_TRACKING_URI
    print(f"MLflow tracking URI: {os.environ['MLFLOW_TRACKING_URI']}")
    
    try:
        # Initialize components
        print("1. Initializing components...")
        data_loader = DataLoader()
        preprocessor = DataPreprocessor()
        trainer = ModelTrainer()
        
        # Load data
        print("2. Loading data...")
        ratings_df, movies_df, users_df = data_loader.load_data()
        print(f"   - Ratings: {ratings_df.shape}")
        print(f"   - Movies: {movies_df.shape}")
        print(f"   - Users: {users_df.shape}")
        
        # Preprocess data
        print("3. Preprocessing data...")
        X, y, preprocessing_artifacts = preprocessor.preprocess_data(
            ratings_df, movies_df, users_df, data_loader.data_root_dir
        )
        print(f"   - Features shape: {X.shape}")
        print(f"   - Target shape: {y.shape}")
        print(f"   - Feature columns: {len(preprocessing_artifacts['feature_columns'])}")
        
        # Fit scaler
        print("4. Fitting scaler...")
        scaler = preprocessor.fit_scaler(X)
        print("   - Scaler fitted successfully")
        
        # Test with Logistic Regression (smaller model for testing)
        print("5. Testing Logistic Regression training...")
        with mlflow.start_run(run_name="Test_Logistic_Regression") as run:
            result = trainer.train_and_evaluate(
                X, y, scaler, preprocessing_artifacts['mlb'], preprocessing_artifacts['ohe_user_features'],
                "LogisticRegression"
            )
            print(f"   - Training completed. Run ID: {run.info.run_id}")
            print(f"   - Accuracy: {result['metrics']['accuracy']:.4f}")
            print(f"   - F1 Score: {result['metrics']['f1_score']:.4f}")
        
        print("✅ All tests passed! Modular pipeline is working correctly.")
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        raise

if __name__ == "__main__":
    test_pipeline() 