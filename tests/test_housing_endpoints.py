"""
Comprehensive test suite for House Price Prediction API endpoints.

Tests cover:
- Health check endpoint
- /predict endpoint (full 18 features)
- /predict-minimal endpoint (flexible 8 features)
- Validation errors and edge cases
- Real predictions using future_unseen_examples.csv

Run with: pytest test_housing_endpoints.py -v
"""

import pytest
import requests
import pandas as pd
import json
import logging
import time
from typing import Dict, Any
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_housing_endpoints.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# API Base URL (adjust if needed)
BASE_URL = "http://localhost:8000"


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def log_api_request(method: str, url: str, data: dict = None, test_name: str = ""):
    """Log API request details."""
    logger.info(f"[{test_name}] {method} {url}")
    if data:
        logger.debug(f"[{test_name}] Request payload: {json.dumps(data, indent=2)}")


def log_api_response(response: requests.Response, test_name: str = "", elapsed_time: float = None):
    """Log API response details."""
    logger.info(f"[{test_name}] Response: {response.status_code} (Time: {elapsed_time:.3f}s)")
    try:
        response_json = response.json()
        logger.debug(f"[{test_name}] Response body: {json.dumps(response_json, indent=2)}")
    except:
        logger.debug(f"[{test_name}] Response body: {response.text}")


def make_request(method: str, url: str, test_name: str, json_data: dict = None) -> tuple:
    """
    Make HTTP request with logging and timing.
    Returns: (response, elapsed_time)
    """
    log_api_request(method, url, json_data, test_name)

    start_time = time.time()
    if method.upper() == "GET":
        response = requests.get(url)
    elif method.upper() == "POST":
        response = requests.post(url, json=json_data)
    else:
        raise ValueError(f"Unsupported HTTP method: {method}")

    elapsed_time = time.time() - start_time
    log_api_response(response, test_name, elapsed_time)

    return response, elapsed_time


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture(scope="module")
def api_url():
    """Base URL for the API."""
    logger.info(f"Using API URL: {BASE_URL}")
    return BASE_URL


@pytest.fixture(scope="module")
def test_data():
    """Load test data from CSV."""
    # Get project root (parent of tests directory)
    project_root = Path(__file__).parent.parent
    csv_path = project_root / "data" / "future_unseen_examples.csv"
    df = pd.read_csv(csv_path, dtype={'zipcode': str})
    return df


@pytest.fixture
def valid_full_house_data():
    """Valid house data with all 18 features."""
    return {
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


@pytest.fixture
def valid_minimal_house_data():
    """Valid house data with 8 core features."""
    return {
        "bedrooms": 3,
        "bathrooms": 2.0,
        "sqft_living": 1500,
        "sqft_lot": 7000,
        "floors": 1.0,
        "sqft_above": 1500,
        "sqft_basement": 0,
        "zipcode": "98103"
    }


# ============================================================================
# HEALTH ENDPOINT TESTS
# ============================================================================

class TestHealthEndpoint:
    """Test suite for /health endpoint."""

    def test_health_endpoint_returns_200(self, api_url):
        """Health endpoint should return 200 OK."""
        logger.info("=" * 60)
        logger.info("TEST: Health endpoint returns 200")

        response, elapsed = make_request("GET", f"{api_url}/health", "test_health_endpoint_returns_200")

        assert response.status_code == 200
        logger.info(f"✓ Test passed (Response time: {elapsed:.3f}s)")

    def test_health_endpoint_structure(self, api_url):
        """Health endpoint should return expected JSON structure."""
        logger.info("=" * 60)
        logger.info("TEST: Health endpoint structure validation")

        response, _ = make_request("GET", f"{api_url}/health", "test_health_endpoint_structure")
        data = response.json()

        assert "status" in data
        assert "model_loaded" in data
        assert "demographics_loaded" in data
        assert "features_count" in data

        logger.info(f"✓ Test passed - All required fields present")

    def test_health_endpoint_model_loaded(self, api_url):
        """Health endpoint should confirm model is loaded."""
        logger.info("=" * 60)
        logger.info("TEST: Health endpoint model loaded verification")

        response, _ = make_request("GET", f"{api_url}/health", "test_health_endpoint_model_loaded")
        data = response.json()

        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data["demographics_loaded"] is True
        assert data["features_count"] == 33  # 8 house + 25 demographic

        logger.info(f"✓ Test passed - Model and demographics loaded, {data['features_count']} features")


# ============================================================================
# /PREDICT ENDPOINT TESTS (Full 18 features)
# ============================================================================

class TestPredictEndpoint:
    """Test suite for /predict endpoint (requires all 18 features)."""

    def test_predict_with_all_fields_success(self, api_url, valid_full_house_data):
        """Predict with all 18 fields should succeed."""
        logger.info("=" * 60)
        logger.info("TEST: /predict with all 18 fields")

        response, elapsed = make_request(
            "POST",
            f"{api_url}/predict",
            "test_predict_with_all_fields_success",
            valid_full_house_data
        )

        assert response.status_code == 200
        data = response.json()

        assert "predicted_price" in data
        assert "zipcode" in data
        assert "model_version" in data
        assert isinstance(data["predicted_price"], (int, float))
        assert data["predicted_price"] > 0

        logger.info(f"✓ Test passed - Predicted price: ${data['predicted_price']:,.2f} (Time: {elapsed:.3f}s)")

    def test_predict_response_structure(self, api_url, valid_full_house_data):
        """Predict response should have correct structure."""
        response = requests.post(
            f"{api_url}/predict",
            json=valid_full_house_data
        )
        data = response.json()

        assert data["zipcode"] == "98103"
        assert data["model_version"] == "baseline_v1"
        # /predict should NOT have defaults_applied
        assert "defaults_applied" not in data or data.get("defaults_applied") is None

    def test_predict_invalid_zipcode(self, api_url, valid_full_house_data):
        """Predict with invalid zipcode should return 400."""
        invalid_data = valid_full_house_data.copy()
        invalid_data["zipcode"] = "99999"  # Not in demographics

        response = requests.post(
            f"{api_url}/predict",
            json=invalid_data
        )

        assert response.status_code == 400
        assert "not found in demographics" in response.json()["detail"].lower()

    def test_predict_invalid_bedrooms_type(self, api_url, valid_full_house_data):
        """Predict with invalid bedrooms type should return 422."""
        invalid_data = valid_full_house_data.copy()
        invalid_data["bedrooms"] = "three"  # Should be int

        response = requests.post(
            f"{api_url}/predict",
            json=invalid_data
        )

        assert response.status_code == 422  # Pydantic validation error

    def test_predict_out_of_range_bedrooms(self, api_url, valid_full_house_data):
        """Predict with out-of-range bedrooms should return 422."""
        invalid_data = valid_full_house_data.copy()
        invalid_data["bedrooms"] = 50  # Max is 20

        response = requests.post(
            f"{api_url}/predict",
            json=invalid_data
        )

        assert response.status_code == 422

    def test_predict_missing_required_field(self, api_url, valid_full_house_data):
        """Predict with missing required field should return 422."""
        incomplete_data = valid_full_house_data.copy()
        del incomplete_data["bedrooms"]  # Remove required field

        response = requests.post(
            f"{api_url}/predict",
            json=incomplete_data
        )

        assert response.status_code == 422


# ============================================================================
# /PREDICT-MINIMAL ENDPOINT TESTS (Flexible 8 features)
# ============================================================================

class TestPredictMinimalEndpoint:
    """Test suite for /predict-minimal endpoint (accepts partial data)."""

    def test_predict_minimal_all_8_fields(self, api_url, valid_minimal_house_data):
        """Predict-minimal with all 8 fields should succeed with no defaults."""
        logger.info("=" * 60)
        logger.info("TEST: /predict-minimal with all 8 fields (no defaults)")

        response, elapsed = make_request(
            "POST",
            f"{api_url}/predict-minimal",
            "test_predict_minimal_all_8_fields",
            valid_minimal_house_data
        )

        assert response.status_code == 200
        data = response.json()

        assert "predicted_price" in data
        assert "defaults_applied" in data
        assert data["defaults_applied"] is None  # No defaults needed

        logger.info(f"✓ Test passed - No defaults needed, Price: ${data['predicted_price']:,.2f}")

    def test_predict_minimal_only_zipcode(self, api_url):
        """Predict-minimal with only zipcode should succeed with all defaults."""
        logger.info("=" * 60)
        logger.info("TEST: /predict-minimal with only zipcode (maximum defaults)")

        minimal_data = {"zipcode": "98103"}

        response, elapsed = make_request(
            "POST",
            f"{api_url}/predict-minimal",
            "test_predict_minimal_only_zipcode",
            minimal_data
        )

        assert response.status_code == 200
        data = response.json()

        assert "predicted_price" in data
        assert "defaults_applied" in data
        assert data["defaults_applied"] is not None
        assert len(data["defaults_applied"]) == 7  # All 7 fields defaulted

        logger.info(f"✓ Test passed - 7 defaults applied: {data['defaults_applied']}")
        logger.info(f"  Predicted price: ${data['predicted_price']:,.2f}")

    def test_predict_minimal_3_fields(self, api_url):
        """Predict-minimal with 3-4 fields should succeed with some defaults."""
        partial_data = {
            "zipcode": "98103",
            "bedrooms": 4,
            "bathrooms": 2.5,
            "sqft_living": 2000
        }

        response = requests.post(
            f"{api_url}/predict-minimal",
            json=partial_data
        )

        assert response.status_code == 200
        data = response.json()

        assert "predicted_price" in data
        assert "defaults_applied" in data
        # Should have defaults for: sqft_lot, floors, sqft_above, sqft_basement
        assert data["defaults_applied"] is not None
        assert len(data["defaults_applied"]) == 4

    def test_predict_minimal_bedrooms_only(self, api_url):
        """Predict-minimal with zipcode + bedrooms should work."""
        partial_data = {
            "zipcode": "98115",
            "bedrooms": 5
        }

        response = requests.post(
            f"{api_url}/predict-minimal",
            json=partial_data
        )

        assert response.status_code == 200
        data = response.json()

        assert data["predicted_price"] > 0
        assert len(data["defaults_applied"]) == 6  # All except bedrooms

    def test_predict_minimal_missing_zipcode(self, api_url):
        """Predict-minimal without zipcode should return 422 (zipcode is required)."""
        invalid_data = {
            "bedrooms": 3,
            "bathrooms": 2.0
        }

        response = requests.post(
            f"{api_url}/predict-minimal",
            json=invalid_data
        )

        assert response.status_code == 422  # Zipcode is required

    def test_predict_minimal_invalid_zipcode(self, api_url):
        """Predict-minimal with invalid zipcode should return 400."""
        invalid_data = {"zipcode": "00000"}

        response = requests.post(
            f"{api_url}/predict-minimal",
            json=invalid_data
        )

        assert response.status_code == 400

    def test_predict_minimal_invalid_data_type(self, api_url):
        """Predict-minimal with invalid data type should return 422."""
        invalid_data = {
            "zipcode": "98103",
            "bedrooms": "five",  # Should be int
            "bathrooms": 2.0
        }

        response = requests.post(
            f"{api_url}/predict-minimal",
            json=invalid_data
        )

        assert response.status_code == 422


# ============================================================================
# REAL DATA TESTS (Using future_unseen_examples.csv)
# ============================================================================

class TestRealPredictions:
    """Test predictions using real data from future_unseen_examples.csv."""

    def test_predict_first_row_from_csv(self, api_url, test_data):
        """Test prediction with first row from CSV."""
        row = test_data.iloc[0]

        house_data = {
            "bedrooms": int(row["bedrooms"]),
            "bathrooms": float(row["bathrooms"]),
            "sqft_living": int(row["sqft_living"]),
            "sqft_lot": int(row["sqft_lot"]),
            "floors": float(row["floors"]),
            "waterfront": int(row["waterfront"]),
            "view": int(row["view"]),
            "condition": int(row["condition"]),
            "grade": int(row["grade"]),
            "sqft_above": int(row["sqft_above"]),
            "sqft_basement": int(row["sqft_basement"]),
            "yr_built": int(row["yr_built"]),
            "yr_renovated": int(row["yr_renovated"]),
            "zipcode": str(row["zipcode"]),
            "lat": float(row["lat"]),
            "long": float(row["long"]),
            "sqft_living15": int(row["sqft_living15"]),
            "sqft_lot15": int(row["sqft_lot15"])
        }

        response = requests.post(
            f"{api_url}/predict",
            json=house_data
        )

        assert response.status_code == 200
        data = response.json()
        assert data["predicted_price"] > 0
        print(f"\nFirst row prediction: ${data['predicted_price']:,.2f}")

    def test_predict_minimal_first_row_from_csv(self, api_url, test_data):
        """Test predict-minimal with core fields from first CSV row."""
        row = test_data.iloc[0]

        minimal_data = {
            "bedrooms": int(row["bedrooms"]),
            "bathrooms": float(row["bathrooms"]),
            "sqft_living": int(row["sqft_living"]),
            "sqft_lot": int(row["sqft_lot"]),
            "floors": float(row["floors"]),
            "sqft_above": int(row["sqft_above"]),
            "sqft_basement": int(row["sqft_basement"]),
            "zipcode": str(row["zipcode"])
        }

        response = requests.post(
            f"{api_url}/predict-minimal",
            json=minimal_data
        )

        assert response.status_code == 200
        data = response.json()
        assert data["predicted_price"] > 0
        assert data["defaults_applied"] is None  # All fields provided
        print(f"\nFirst row minimal prediction: ${data['predicted_price']:,.2f}")

    def test_batch_predictions_sample(self, api_url, test_data):
        """Test predictions on sample of 10 rows from CSV."""
        logger.info("=" * 60)
        logger.info("TEST: Batch predictions on 10 CSV rows")

        sample_rows = test_data.head(10)
        predictions = []
        total_time = 0

        for idx, row in sample_rows.iterrows():
            minimal_data = {
                "bedrooms": int(row["bedrooms"]),
                "bathrooms": float(row["bathrooms"]),
                "sqft_living": int(row["sqft_living"]),
                "sqft_lot": int(row["sqft_lot"]),
                "floors": float(row["floors"]),
                "sqft_above": int(row["sqft_above"]),
                "sqft_basement": int(row["sqft_basement"]),
                "zipcode": str(row["zipcode"])
            }

            response, elapsed = make_request(
                "POST",
                f"{api_url}/predict-minimal",
                f"batch_prediction_row_{idx}",
                minimal_data
            )

            assert response.status_code == 200
            data = response.json()
            predictions.append(data["predicted_price"])
            total_time += elapsed

        assert len(predictions) == 10
        assert all(p > 0 for p in predictions)

        logger.info(f"✓ Test passed - 10 predictions completed")
        logger.info(f"  Min price: ${min(predictions):,.2f}")
        logger.info(f"  Max price: ${max(predictions):,.2f}")
        logger.info(f"  Avg price: ${sum(predictions)/len(predictions):,.2f}")
        logger.info(f"  Total time: {total_time:.3f}s")
        logger.info(f"  Avg time per prediction: {total_time/10:.3f}s")

    def test_predict_all_csv_rows(self, api_url, test_data):
        """Test predictions on all 100 rows from CSV (comprehensive test)."""
        logger.info("=" * 60)
        logger.info("TEST: Comprehensive - Predict all 100 CSV rows")
        logger.info(f"Starting comprehensive test at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        successful_predictions = 0
        failed_predictions = 0
        predictions = []
        total_time = 0
        failures = []

        for idx, row in test_data.iterrows():
            minimal_data = {
                "bedrooms": int(row["bedrooms"]),
                "bathrooms": float(row["bathrooms"]),
                "sqft_living": int(row["sqft_living"]),
                "sqft_lot": int(row["sqft_lot"]),
                "floors": float(row["floors"]),
                "sqft_above": int(row["sqft_above"]),
                "sqft_basement": int(row["sqft_basement"]),
                "zipcode": str(row["zipcode"])
            }

            response, elapsed = make_request(
                "POST",
                f"{api_url}/predict-minimal",
                f"comprehensive_row_{idx}",
                minimal_data
            )
            total_time += elapsed

            if response.status_code == 200:
                successful_predictions += 1
                data = response.json()
                predictions.append(data["predicted_price"])
            else:
                failed_predictions += 1
                error_msg = f"Row {idx}: {response.status_code} - {response.json()}"
                logger.error(error_msg)
                failures.append(error_msg)

            # Log progress every 20 rows
            if (idx + 1) % 20 == 0:
                logger.info(f"  Progress: {idx + 1}/100 rows processed ({successful_predictions} success, {failed_predictions} failed)")

        logger.info("=" * 60)
        logger.info("COMPREHENSIVE TEST RESULTS:")
        logger.info(f"  Total rows: {len(test_data)}")
        logger.info(f"  Successful: {successful_predictions} ({successful_predictions/len(test_data)*100:.1f}%)")
        logger.info(f"  Failed: {failed_predictions} ({failed_predictions/len(test_data)*100:.1f}%)")
        logger.info(f"  Total time: {total_time:.3f}s")
        logger.info(f"  Avg time per prediction: {total_time/len(test_data):.3f}s")

        if predictions:
            logger.info(f"  Min price: ${min(predictions):,.2f}")
            logger.info(f"  Max price: ${max(predictions):,.2f}")
            logger.info(f"  Avg price: ${sum(predictions)/len(predictions):,.2f}")

        if failures:
            logger.error(f"FAILURES ({len(failures)}):")
            for failure in failures:
                logger.error(f"  {failure}")

        assert successful_predictions > 90, f"Success rate too low: {successful_predictions}/100"
        assert failed_predictions < 10, f"Too many failures: {failed_predictions}/100"

        logger.info(f"✓ Comprehensive test passed!")
        logger.info("=" * 60)


# ============================================================================
# EDGE CASES AND ERROR HANDLING
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_predict_minimal_extreme_values(self, api_url):
        """Test with extreme but valid values."""
        extreme_data = {
            "zipcode": "98103",
            "bedrooms": 20,  # Max allowed
            "bathrooms": 10.0,  # Max allowed
            "sqft_living": 15000,  # Max allowed
            "sqft_lot": 1000000,  # Max allowed
            "floors": 5.0,  # Max allowed
            "sqft_above": 15000,
            "sqft_basement": 0
        }

        response = requests.post(
            f"{api_url}/predict-minimal",
            json=extreme_data
        )

        assert response.status_code == 200

    def test_predict_minimal_minimum_values(self, api_url):
        """Test with minimum valid values."""
        minimum_data = {
            "zipcode": "98103",
            "bedrooms": 0,  # Min allowed
            "bathrooms": 0.0,  # Min allowed
            "sqft_living": 100,  # Min allowed
            "sqft_lot": 0,  # Min allowed
            "floors": 1.0,  # Min allowed
            "sqft_above": 0,
            "sqft_basement": 0
        }

        response = requests.post(
            f"{api_url}/predict-minimal",
            json=minimum_data
        )

        assert response.status_code == 200

    def test_empty_request_body(self, api_url):
        """Test with empty request body."""
        response = requests.post(
            f"{api_url}/predict-minimal",
            json={}
        )

        assert response.status_code == 422  # Missing required zipcode

    def test_null_values(self, api_url):
        """Test with null values in optional fields."""
        data_with_nulls = {
            "zipcode": "98103",
            "bedrooms": None,  # Optional in predict-minimal
            "bathrooms": None,
            "sqft_living": None,
            "sqft_lot": None,
            "floors": None,
            "sqft_above": None,
            "sqft_basement": None
        }

        response = requests.post(
            f"{api_url}/predict-minimal",
            json=data_with_nulls
        )

        assert response.status_code == 200  # Should use defaults for all


# ============================================================================
# PYTEST HOOKS FOR LOGGING
# ============================================================================

def pytest_sessionstart(session):
    """Called at the start of test session."""
    logger.info("=" * 80)
    logger.info("HOUSE PRICE API TEST SUITE - SESSION START")
    logger.info(f"Session started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"API Base URL: {BASE_URL}")
    logger.info("=" * 80)


def pytest_sessionfinish(session, exitstatus):
    """Called at the end of test session."""
    logger.info("=" * 80)
    logger.info("HOUSE PRICE API TEST SUITE - SESSION END")
    logger.info(f"Session finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Exit status: {exitstatus} (0=success, 1=tests failed, 2=interrupted)")
    logger.info("=" * 80)


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    # Can be run directly: python test_housing_endpoints.py
    # Or with pytest: pytest test_housing_endpoints.py -v
    logger.info("Starting test execution...")
    pytest.main([__file__, "-v", "--tb=short"])
