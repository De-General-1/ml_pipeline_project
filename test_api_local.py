#!/usr/bin/env python3
"""
Test the inference API locally with sample data
"""

import requests
import json

def test_api():
    base_url = "http://localhost:8000"
    
    # Test data samples
    test_cases = [
        {
            "name": "Young male student likes action movies",
            "data": {
                "user_id": 123,
                "movie_id": 456,
                "age": 22,
                "gender": "M",
                "occupation": "student",
                "genres": ["Action", "Adventure"],
                "release_year": 1995
            }
        },
        {
            "name": "Middle-aged female teacher likes romantic comedies",
            "data": {
                "user_id": 789,
                "movie_id": 101,
                "age": 35,
                "gender": "F",
                "occupation": "educator",
                "genres": ["Comedy", "Romance"],
                "release_year": 1998
            }
        },
        {
            "name": "Older male engineer likes sci-fi",
            "data": {
                "user_id": 555,
                "movie_id": 777,
                "age": 45,
                "gender": "M",
                "occupation": "engineer",
                "genres": ["Sci-Fi", "Thriller"],
                "release_year": 1999
            }
        }
    ]
    
    print("🧪 Testing Inference API locally...")
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health")
        print(f"\n✅ Health Check: {response.status_code}")
        print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return
    
    # Test predictions
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🎬 Test Case {i}: {test_case['name']}")
        print(f"   Input: {json.dumps(test_case['data'], indent=2)}")
        
        try:
            response = requests.post(
                f"{base_url}/predict",
                json=test_case['data'],
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ Status: {response.status_code}")
                print(f"   📊 Engagement Probability: {result['engagement_probability']:.3f}")
                print(f"   🎯 Prediction: {'Will Engage' if result['engagement_prediction'] == 1 else 'Will Not Engage'}")
                print(f"   🤖 Model: {result['model_version']}")
            else:
                print(f"   ❌ Status: {response.status_code}")
                print(f"   Error: {response.text}")
                
        except Exception as e:
            print(f"   ❌ Request failed: {e}")
    
    print("\n🏁 Testing completed!")

if __name__ == "__main__":
    test_api()