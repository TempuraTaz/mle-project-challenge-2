"""
Simple test suite for House Price Prediction API.

Tests cover the essential functionality:
- Health check endpoint
- Successful predictions
- Error handling (invalid data, missing fields)
- Predict-minimal with defaults

Run with: pytest tests/test_api.py -v
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
from fastapi.testclient import TestClient
from src.api import app


@pytest.fixture(scope="module")
def client():
    """Create test client with startup/shutdown events."""
    with TestClient(app) as test_client:
        yield test_client


# ============================================================================
# HEALTH CHECK TESTS
# ============================================================================

def test_health_check(client):
    """Health endpoint should return 200 with healthy status."""
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["demographics_loaded"] is True


def test_root_endpoint(client):
    """Root endpoint should return API info."""
    response = client.get("/")
    assert response.status_code == 200

    data = response.json()
    assert "message" in data
    assert "version" in data


# ============================================================================
# PREDICTION TESTS
# ============================================================================

def test_predict_success(client):
    """Valid prediction with all fields should return predicted price."""
    house_data = {
        "bedrooms": 3,
        "bathrooms": 2.0,
        "sqft_living": 1500,
        "sqft_lot": 7000,
        "floors": 1.0,
        "waterfront": 0,
        "view": 0,
        "condition": 3,
        "grade": 7,
        "sqft_above": 1500,
        "sqft_basement": 0,
        "yr_built": 1970,
        "yr_renovated": 0,
        "zipcode": "98103",
        "lat": 47.6692,
        "long": -122.3419,
        "sqft_living15": 1440,
        "sqft_lot15": 7000
    }

    response = client.post("/predict", json=house_data)
    assert response.status_code == 200

    data = response.json()
    assert "predicted_price" in data
    assert "zipcode" in data
    assert "model_version" in data
    assert data["predicted_price"] > 0
    assert data["zipcode"] == "98103"


def test_predict_invalid_zipcode(client):
    """Prediction with invalid zipcode should return 400."""
    house_data = {
        "bedrooms": 3,
        "bathrooms": 2.0,
        "sqft_living": 1500,
        "sqft_lot": 7000,
        "floors": 1.0,
        "waterfront": 0,
        "view": 0,
        "condition": 3,
        "grade": 7,
        "sqft_above": 1500,
        "sqft_basement": 0,
        "yr_built": 1970,
        "yr_renovated": 0,
        "zipcode": "99999",  # Invalid zipcode
        "lat": 47.6692,
        "long": -122.3419,
        "sqft_living15": 1440,
        "sqft_lot15": 7000
    }

    response = client.post("/predict", json=house_data)
    assert response.status_code == 400
    assert "not found" in response.json()["detail"].lower()


def test_predict_missing_required_field(client):
    """Prediction with missing required field should return 422."""
    incomplete_data = {
        "bathrooms": 2.0,
        "sqft_living": 1500,
        # Missing bedrooms and other required fields
        "zipcode": "98103"
    }

    response = client.post("/predict", json=incomplete_data)
    assert response.status_code == 422


def test_predict_invalid_data_type(client):
    """Prediction with invalid data type should return 422."""
    invalid_data = {
        "bedrooms": "three",  # Should be int
        "bathrooms": 2.0,
        "sqft_living": 1500,
        "sqft_lot": 7000,
        "floors": 1.0,
        "waterfront": 0,
        "view": 0,
        "condition": 3,
        "grade": 7,
        "sqft_above": 1500,
        "sqft_basement": 0,
        "yr_built": 1970,
        "yr_renovated": 0,
        "zipcode": "98103",
        "lat": 47.6692,
        "long": -122.3419,
        "sqft_living15": 1440,
        "sqft_lot15": 7000
    }

    response = client.post("/predict", json=invalid_data)
    assert response.status_code == 422


# ============================================================================
# PREDICT-MINIMAL TESTS
# ============================================================================

def test_predict_minimal_with_all_fields(client):
    """Predict-minimal with all 18 fields should not apply defaults."""
    house_data = {
        "bedrooms": 3,
        "bathrooms": 2.0,
        "sqft_living": 1500,
        "sqft_lot": 7000,
        "floors": 1.0,
        "waterfront": 0,
        "view": 0,
        "condition": 3,
        "grade": 7,
        "sqft_above": 1500,
        "sqft_basement": 0,
        "yr_built": 1970,
        "yr_renovated": 0,
        "zipcode": "98103",
        "lat": 47.6692,
        "long": -122.3419,
        "sqft_living15": 1440,
        "sqft_lot15": 7000
    }

    response = client.post("/predict-minimal", json=house_data)
    assert response.status_code == 200

    data = response.json()
    assert "predicted_price" in data
    assert "defaults_applied" in data
    assert data["defaults_applied"] is None  # No defaults needed


def test_predict_minimal_only_zipcode(client):
    """Predict-minimal with only zipcode should apply all defaults."""
    minimal_data = {"zipcode": "98103"}

    response = client.post("/predict-minimal", json=minimal_data)
    assert response.status_code == 200

    data = response.json()
    assert "predicted_price" in data
    assert "defaults_applied" in data
    assert data["defaults_applied"] is not None
    assert len(data["defaults_applied"]) == 17  # All 17 house fields defaulted


def test_predict_minimal_partial_data(client):
    """Predict-minimal with some fields should apply some defaults."""
    partial_data = {
        "zipcode": "98103",
        "bedrooms": 4,
        "bathrooms": 2.5,
        "sqft_living": 2000
    }

    response = client.post("/predict-minimal", json=partial_data)
    assert response.status_code == 200

    data = response.json()
    assert "predicted_price" in data
    assert "defaults_applied" in data
    assert data["defaults_applied"] is not None
    assert len(data["defaults_applied"]) == 14  # 14 fields defaulted (18 total - 4 provided)


def test_predict_minimal_missing_zipcode(client):
    """Predict-minimal without zipcode should return 422."""
    invalid_data = {
        "bedrooms": 3,
        "bathrooms": 2.0
    }

    response = client.post("/predict-minimal", json=invalid_data)
    assert response.status_code == 422  # Zipcode is required