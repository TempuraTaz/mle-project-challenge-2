"""
Centralized Configuration for House Price Prediction Project

This module contains all configuration constants used across the project:
- File paths (data, model artifacts)
- Model parameters
- Training parameters
- API settings

Usage:
    from src.config import Config
    model = pickle.load(open(Config.MODEL_PATH, 'rb'))
"""

from pathlib import Path

# Project root directory (parent of src/)
PROJECT_ROOT = Path(__file__).parent.parent


class Config:
    """Centralized configuration for the house price prediction project"""

    # ========================================================================
    # DATA PATHS
    # ========================================================================
    DATA_DIR = PROJECT_ROOT / "data"
    SALES_DATA_PATH = DATA_DIR / "kc_house_data.csv"
    DEMOGRAPHICS_PATH = DATA_DIR / "zipcode_demographics.csv"
    FUTURE_EXAMPLES_PATH = DATA_DIR / "future_unseen_examples.csv"

    # ========================================================================
    # MODEL PATHS
    # ========================================================================
    MODEL_DIR = PROJECT_ROOT / "model"
    MODEL_PATH = MODEL_DIR / "model_v5.pkl"
    MODEL_FEATURES_PATH = MODEL_DIR / "model_features_v5.json"
    DEFAULTS_PATH = MODEL_DIR / "defaults_v5.json"

    # ========================================================================
    # MODEL CONFIGURATION
    # ========================================================================
    MODEL_VERSION = "xgboost_v5"  

    # Feature selection for training (house features only, demographics joined later)
    SALES_COLUMN_SELECTION = [
        'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
        'sqft_above', 'sqft_basement', 'zipcode'
    ]

    # House features (used for data loading - excludes 'price' and 'zipcode')
    HOUSE_FEATURES = [
        'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
        'sqft_above', 'sqft_basement'
    ]

    # ========================================================================
    # TRAINING PARAMETERS
    # ========================================================================
    RANDOM_STATE = 42  # For reproducibility
    TEST_SIZE = 0.25   # Train/test split ratio

    # ========================================================================
    # API CONFIGURATION
    # ========================================================================
    API_TITLE = "House Price Prediction API"
    API_DESCRIPTION = "API for predicting house prices in Seattle area"
    API_VERSION = "1.0.0"
    API_HOST = "0.0.0.0"
    API_PORT = 8000


class LogConfig:
    """Logging configuration"""
    LEVEL = "INFO"
    FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


# Validate that required directories exist
def validate_paths():
    """Validate that required data and model directories exist"""
    if not Config.DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory not found: {Config.DATA_DIR}")

    if not Config.MODEL_DIR.exists():
        raise FileNotFoundError(f"Model directory not found: {Config.MODEL_DIR}")


if __name__ == "__main__":
    # Quick validation when run directly
    print("Configuration Validation")
    print("=" * 60)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Data Directory: {Config.DATA_DIR} (exists: {Config.DATA_DIR.exists()})")
    print(f"Model Directory: {Config.MODEL_DIR} (exists: {Config.MODEL_DIR.exists()})")
    print(f"Model Version: {Config.MODEL_VERSION}")
    print("=" * 60)

    try:
        validate_paths()
        print("✓ All paths validated successfully")
    except FileNotFoundError as e:
        print(f"✗ Validation failed: {e}")
