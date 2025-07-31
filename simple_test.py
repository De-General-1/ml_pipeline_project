#!/usr/bin/env python3

import os
import joblib

print("Testing basic imports and preprocessing artifacts...")

try:
    # Test loading preprocessing artifacts
    print("Loading preprocessing artifacts...")
    scaler = joblib.load("models/scaler.joblib")
    mlb = joblib.load("models/multilabel_binarizer.joblib")
    ohe_user_features = joblib.load("models/ohe_user.joblib")
    print("Preprocessing artifacts loaded successfully!")
    
    print("Basic test passed!")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc() 