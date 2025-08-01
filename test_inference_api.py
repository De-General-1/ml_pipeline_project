#!/usr/bin/env python3
"""
Test script for the inference API
"""

import requests
import json

def test_inference_api(base_url="http://localhost:8000"):
    """Test the inference API endpoints"""
    
    print(f"🧪 Testing Inference API at {base_url}")
    
    # Test health check
    print("\n1. Testing health check...")
    try:
        response = requests.get(f"{base_url}/health")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test model info
    print("\n2. Testing model info...")
    try:
        response = requests.get(f"{base_url}/model-info")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test prediction
    print("\n3. Testing prediction...")
    test_data = {
        "user_id": 123,
        "movie_id": 456,
        "age": 25,
        "gender": "M",
        "occupation": "student",
        "genres": ["Action", "Comedy"],
        "release_year": 1995
    }
    
    try:
        response = requests.post(
            f"{base_url}/predict",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        print(f"   Status: {response.status_code}")
        print(f"   Request: {json.dumps(test_data, indent=2)}")
        print(f"   Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\nAPI testing completed!")

if __name__ == "__main__":
    import sys
    
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    test_inference_api(base_url)