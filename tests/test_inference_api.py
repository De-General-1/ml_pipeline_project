import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from api.inference_api import app

client = TestClient(app)

@pytest.fixture
def mock_model_artifacts():
    """Mock model artifacts for testing"""
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = [[0.3, 0.7]]
    mock_model.predict.return_value = [1]
    
    mock_scaler = MagicMock()
    mock_scaler.transform.return_value = [[1995.0]]
    
    mock_mlb = MagicMock()
    mock_mlb.classes_ = ['Action', 'Comedy', 'Drama']
    mock_mlb.transform.return_value = [[1, 1, 0]]
    
    mock_ohe = MagicMock()
    mock_ohe.get_feature_names_out.return_value = ['Gender_F', 'Gender_M', 'Age_25', 'Occupation_student']
    mock_ohe.transform.return_value = [[0, 1, 1, 1]]
    
    return mock_model, mock_scaler, mock_mlb, mock_ohe

def test_root_endpoint():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "Movie Engagement Prediction API"

def test_health_endpoint():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()

@patch('api.inference_api.boto3.Session')
@patch('api.inference_api.model', None)
def test_predict_endpoint_model_not_loaded(mock_session):
    """Test prediction when model is not loaded"""
    test_data = {
        "user_id": 123,
        "movie_id": 456,
        "age": 25,
        "gender": "M",
        "occupation": "student",
        "genres": ["Action", "Comedy"],
        "release_year": 1995
    }
    
    response = client.post("/predict", json=test_data)
    assert response.status_code == 503

@patch('api.inference_api.model')
@patch('api.inference_api.scaler')
@patch('api.inference_api.mlb')
@patch('api.inference_api.ohe_user')
@patch('api.inference_api.feature_columns')
def test_predict_endpoint_success(mock_features, mock_ohe, mock_mlb, mock_scaler, mock_model):
    """Test successful prediction"""
    # Setup mocks
    mock_model.predict_proba.return_value = [[0.3, 0.7]]
    mock_model.predict.return_value = [1]
    mock_scaler.transform.return_value = [[1995.0]]
    mock_mlb.classes_ = ['Action', 'Comedy', 'Drama']
    mock_mlb.transform.return_value = [[1, 1, 0]]
    mock_ohe.get_feature_names_out.return_value = ['Gender_F', 'Gender_M', 'Age_25', 'Occupation_student']
    mock_ohe.transform.return_value = [[0, 1, 1, 1]]
    mock_features.__iter__ = lambda x: iter(['ReleaseYear', 'Action', 'Comedy', 'Drama', 'Gender_F', 'Gender_M', 'Age_25', 'Occupation_student'])
    
    test_data = {
        "user_id": 123,
        "movie_id": 456,
        "age": 25,
        "gender": "M",
        "occupation": "student",
        "genres": ["Action", "Comedy"],
        "release_year": 1995
    }
    
    response = client.post("/predict", json=test_data)
    assert response.status_code == 200
    
    result = response.json()
    assert result["user_id"] == 123
    assert result["movie_id"] == 456
    assert "engagement_probability" in result
    assert "engagement_prediction" in result

def test_predict_endpoint_invalid_data():
    """Test prediction with invalid data"""
    test_data = {
        "user_id": "invalid",  # Should be int
        "movie_id": 456,
        "age": 25,
        "gender": "M",
        "occupation": "student",
        "genres": ["Action"],
        "release_year": 1995
    }
    
    response = client.post("/predict", json=test_data)
    assert response.status_code == 422  # Validation error