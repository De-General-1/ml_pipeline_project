#!/usr/bin/env python3
"""
Test the model directly without the API
"""

import boto3
import pickle
import pandas as pd
import numpy as np
import os

def test_model_prediction():
    """Test model prediction directly"""
    print("🧪 Testing model prediction directly...")
    
    # Set AWS profile
    os.environ['AWS_PROFILE'] = 'degen-mlops'
    
    try:
        # Download model from S3
        session = boto3.Session(profile_name='degen-mlops')
        s3_client = session.client('s3')
        
        bucket = 'phase3-mlops-source-bucket-degen-1'
        model_path = 'models/randomforestclassifier'
        
        artifacts = ['model.pkl', 'scaler.pkl', 'multilabel_binarizer.pkl', 'ohe_user.pkl']
        
        print("📥 Downloading model artifacts from S3...")
        for artifact in artifacts:
            s3_key = f"{model_path}/{artifact}"
            local_path = f"/tmp/{artifact}"
            print(f"   Downloading {s3_key}...")
            s3_client.download_file(bucket, s3_key, local_path)
        
        # Load artifacts
        print("🔧 Loading model artifacts...")
        with open('/tmp/model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        with open('/tmp/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        with open('/tmp/multilabel_binarizer.pkl', 'rb') as f:
            mlb = pickle.load(f)
        
        with open('/tmp/ohe_user.pkl', 'rb') as f:
            ohe_user = pickle.load(f)
        
        print("✅ Model artifacts loaded successfully!")
        print(f"   Model type: {type(model).__name__}")
        print(f"   Genre classes: {len(mlb.classes_)}")
        print(f"   User features: {len(ohe_user.get_feature_names_out(['Gender', 'Age', 'Occupation']))}")
        
        # Test cases
        test_cases = [
            {
                "name": "Young male student likes action movies",
                "age": 22,
                "gender": "M",
                "occupation": "student",
                "genres": ["Action", "Adventure"],
                "release_year": 1995
            },
            {
                "name": "Middle-aged female teacher likes romantic comedies",
                "age": 35,
                "gender": "F",
                "occupation": "educator",
                "genres": ["Comedy", "Romance"],
                "release_year": 1998
            }
        ]
        
        # Make predictions
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n🎬 Test Case {i}: {test_case['name']}")
            
            # Create feature vector
            data = {
                'ReleaseYear': [test_case['release_year']],
                'Gender': [test_case['gender']],
                'Age': [test_case['age']],
                'Occupation': [test_case['occupation']]
            }
            
            df = pd.DataFrame(data)
            
            # Process genres
            genre_features = mlb.transform([test_case['genres']])
            genre_df = pd.DataFrame(genre_features, columns=mlb.classes_)
            
            # Process user features
            user_features = ohe_user.transform(df[['Gender', 'Age', 'Occupation']])
            user_df = pd.DataFrame(user_features, columns=ohe_user.get_feature_names_out(['Gender', 'Age', 'Occupation']))
            
            # Combine features
            feature_df = pd.concat([df[['ReleaseYear']], genre_df, user_df], axis=1)
            
            # Scale release year
            feature_df['ReleaseYear'] = scaler.transform(feature_df[['ReleaseYear']])
            
            # Make prediction
            prediction_proba = model.predict_proba(feature_df)[0]
            prediction = model.predict(feature_df)[0]
            
            print(f"   📊 Engagement Probability: {prediction_proba[1]:.3f}")
            print(f"   🎯 Prediction: {'Will Engage' if prediction == 1 else 'Will Not Engage'}")
            print(f"   📈 Confidence: {max(prediction_proba):.3f}")
        
        print("\n✅ Model testing completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model_prediction()