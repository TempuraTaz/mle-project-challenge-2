"""
Simple API validation script.

Tests the API with real data from future_unseen_examples.csv.
Tests both /predict and /predict-minimal endpoints.

Usage: python validate_api.py
(Make sure API is running on localhost:8000)
"""

import requests
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S %f'
)
logger = logging.getLogger(__name__)

# Configuration
API_URL = "http://localhost:8000"
DATA_FILE = Path(__file__).parent.parent / "data" / "future_unseen_examples.csv"
NUM_SAMPLES = 100  


def test_predict_endpoint(df):
    """Test /predict endpoint with full house data."""
    logger.info("="*70)
    logger.info("TESTING /predict ENDPOINT (Full data)")
    logger.info("="*70)

    success_count = 0

    for idx in range(min(NUM_SAMPLES, len(df))):
        row = df.iloc[idx]

        # Build full payload
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

        try:
            response = requests.post(f"{API_URL}/predict", json=house_data)

            if response.status_code == 200:
                data = response.json()
                success_count += 1
                logger.info(f"Row {idx + 1}: ✓ SUCCESS | Zipcode: {data['zipcode']} | Price: ${data['predicted_price']:,.2f}")
            else:
                logger.error(f"Row {idx + 1}: ✗ FAILED | Status: {response.status_code} | Error: {response.json()['detail']}")

        except Exception as e:
            logger.error(f"Row {idx + 1}: ✗ ERROR | {str(e)}")

    logger.info("-"*70)
    logger.info(f"RESULTS: {success_count}/{NUM_SAMPLES} successful predictions")
    logger.info("="*70)


def test_predict_minimal_endpoint():
    """Test /predict-minimal endpoint with mock partial data."""
    logger.info("="*70)
    logger.info("TESTING /predict-minimal ENDPOINT (Partial data)")
    logger.info("="*70)

    # Mock data - different levels of completeness
    test_cases = [
        {
            "name": "Only zipcode",
            "data": {"zipcode": "98103"}
        },
        {
            "name": "Zipcode + 3 fields",
            "data": {
                "zipcode": "98115",
                "bedrooms": 4,
                "bathrooms": 2.5,
                "sqft_living": 2500
            }
        },
        {
            "name": "Zipcode + 7 fields",
            "data": {
                "zipcode": "98117",
                "bedrooms": 3,
                "bathrooms": 2.0,
                "sqft_living": 1800,
                "sqft_lot": 5000,
                "floors": 2.0,
                "sqft_above": 1800,
                "sqft_basement": 0
            }
        }
    ]

    success_count = 0

    for i, test_case in enumerate(test_cases, 1):
        try:
            response = requests.post(f"{API_URL}/predict-minimal", json=test_case["data"])

            if response.status_code == 200:
                data = response.json()
                success_count += 1

                defaults_count = len(data["defaults_applied"]) if data["defaults_applied"] else 0
                logger.info(f"Test {i} ({test_case['name']}): ✓ SUCCESS | Zipcode: {data['zipcode']} | Price: ${data['predicted_price']:,.2f} | Defaults: {defaults_count} fields")
            else:
                logger.error(f"Test {i} ({test_case['name']}): ✗ FAILED | Status: {response.status_code} | Error: {response.json()['detail']}")

        except Exception as e:
            logger.error(f"Test {i} ({test_case['name']}): ✗ ERROR | {str(e)}")

    logger.info("-"*70)
    logger.info(f"RESULTS: {success_count}/{len(test_cases)} successful predictions")
    logger.info("="*70)


def check_api_health():
    """Check if API is running and healthy."""
    print("\n" + "="*70)
    print("CHECKING API HEALTH")
    print("="*70)

    try:
        response = requests.get(f"{API_URL}/health", timeout=5)

        if response.status_code == 200:
            data = response.json()
            print(f"\n✓ API is {data['status'].upper()}")
            print(f"  Model Loaded: {data['model_loaded']}")
            print(f"  Demographics Loaded: {data['demographics_loaded']}")
            print(f"  Features Count: {data['features_count']}")
            return True
        else:
            print(f"\n✗ API returned status {response.status_code}")
            return False

    except requests.exceptions.ConnectionError:
        print(f"\n✗ Cannot connect to API at {API_URL}")
        print("  Make sure the API is running: uvicorn src.api:app --reload")
        return False
    except Exception as e:
        print(f"\n✗ Error checking API health: {str(e)}")
        return False


def main():
    """Main validation function."""
    print("\n" + "#"*70)
    print("# API VALIDATION SCRIPT")
    print("#"*70)

    # Step 1: Check API health
    if not check_api_health():
        print("\n✗ API is not healthy. Exiting.")
        return

    # Step 2: Load data
    print(f"\n{'='*70}")
    print("LOADING DATA")
    print("="*70)

    if not DATA_FILE.exists():
        print(f"\n✗ Data file not found: {DATA_FILE}")
        return

    df = pd.read_csv(DATA_FILE, dtype={'zipcode': str})
    print(f"\n✓ Loaded {len(df)} rows from {DATA_FILE.name}")
    print(f"  Testing {NUM_SAMPLES} samples")

    # Step 3: Test endpoints
    test_predict_endpoint(df)
    test_predict_minimal_endpoint()

    # Final summary
    print("\n" + "#"*70)
    print("# VALIDATION COMPLETE")
    print("#"*70)
    print()


if __name__ == "__main__":
    main()