"""
Model Evaluation Script - V3

Evaluates the v3 house price prediction model performance and generalization.
Uses proper train/validation/test split to assess model appropriateness.
"""

import pickle
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration - file paths
MODEL_PATH = Path("model/model_v3.pkl")
MODEL_FEATURES_PATH = Path("model/model_features_v3.json")
SALES_DATA_PATH = Path("data/kc_house_data.csv")
DEMOGRAPHICS_PATH = Path("data/zipcode_demographics.csv")

# Configuration - training parameters
RANDOM_STATE = 42  # Must match create_model_v3.py


def load_data():
    """Load and prepare data using the same process as training"""
    try:
        # Load model features to know which columns to select
        logger.info(f"Loading model features from {MODEL_FEATURES_PATH}")
        with open(MODEL_FEATURES_PATH, 'r') as f:
            model_features = json.load(f)

        # Extract house features (non-demographic features)
        # Demographics are merged later, so we only need house features from CSV
        house_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
                         'floors', 'waterfront', 'view', 'condition', 'grade',
                         'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated',
                         'lat', 'long', 'sqft_living15', 'sqft_lot15', 'zipcode']

        # Add 'price' target variable
        columns_to_load = ['price'] + house_features

        logger.info(f"Loading sales data from {SALES_DATA_PATH}")
        sales_data = pd.read_csv(SALES_DATA_PATH,
                                usecols=columns_to_load,
                                dtype={'zipcode': str})

        logger.info(f"Loading demographics from {DEMOGRAPHICS_PATH}")
        demographics = pd.read_csv(DEMOGRAPHICS_PATH,
                                  dtype={'zipcode': str})

        # Merge data same as training
        logger.info("Merging sales data with demographics")
        merged_data = sales_data.merge(demographics, how="left", on="zipcode").drop(columns="zipcode")

        y = merged_data.pop('price')
        X = merged_data

        logger.info(f"Data loaded: {len(X)} samples, {len(X.columns)} features")
        return X, y

    except FileNotFoundError as e:
        logger.error(f"Required file not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def load_model():
    """Load the trained model"""
    try:
        logger.info(f"Loading trained model from {MODEL_PATH}")
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        logger.info("Model loaded successfully")
        return model

    except FileNotFoundError:
        logger.error(f"Model file not found at {MODEL_PATH}")
        logger.error("Please run create_model_v3.py first to train the model")
        raise
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def calculate_metrics(y_true, y_pred):
    """Calculate regression metrics including MAPE"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return rmse, mae, r2, mape


def main():
    """Run model evaluation and generate report"""
    try:
        logger.info("=" * 60)
        logger.info("MODEL EVALUATION REPORT - V3")
        logger.info("=" * 60)

        # Load data and model
        X, y = load_data()
        model = load_model()

        # Recreate the EXACT same split used during model training
        # This ensures we evaluate on the same test set the model was designed for
        logger.info(f"Splitting data with random_state={RANDOM_STATE}")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=RANDOM_STATE)

        logger.info(f"Dataset: {len(X)} samples")
        logger.info(f"Train: {len(X_train)} | Test: {len(X_test)}")
        logger.info("Note: Using same train/test split as original model training")

        # Generate predictions
        logger.info("Generating predictions...")
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Calculate metrics for each set
        train_rmse, train_mae, train_r2, train_mape = calculate_metrics(y_train, y_pred_train)
        test_rmse, test_mae, test_r2, test_mape = calculate_metrics(y_test, y_pred_test)

        # Display results
        logger.info("")
        logger.info("PERFORMANCE METRICS:")
        logger.info("-" * 55)
        logger.info(f"{'Set':<12} {'RMSE':<12} {'MAE':<12} {'R²':<8} {'MAPE':<8}")
        logger.info("-" * 55)
        logger.info(f"{'Train':<12} ${train_rmse:<11,.0f} ${train_mae:<11,.0f} {train_r2:<8.3f} {train_mape:<7.1f}%")
        logger.info(f"{'Test':<12} ${test_rmse:<11,.0f} ${test_mae:<11,.0f} {test_r2:<8.3f} {test_mape:<7.1f}%")

        # Overfitting analysis
        train_test_gap = train_r2 - test_r2

        logger.info("")
        logger.info("GENERALIZATION ANALYSIS:")
        logger.info(f"Train-Test R² gap: {train_test_gap:.3f}")

        if train_test_gap > 0.15:
            logger.warning("Significant overfitting detected (gap > 0.15)")
        elif train_test_gap > 0.10:
            logger.info("Moderate overfitting (gap > 0.10)")
        else:
            logger.info("Good generalization (gap <= 0.10)")

        # Business context
        median_price = y_test.median()

        logger.info("")
        logger.info("BUSINESS CONTEXT:")
        logger.info(f"Median house price: ${median_price:,.0f}")
        logger.info(f"Average prediction error: {test_mape:.1f}% (MAPE)")
        logger.info(f"Typical error on median house: ${median_price * test_mape/100:,.0f}")

        # Model assessment
        logger.info("")
        logger.info("MODEL ASSESSMENT:")
        logger.info(f"Explains {test_r2:.1%} of price variance on unseen data")

        if test_r2 > 0.8:
            logger.info("✓ Excellent model performance (R² > 0.80)")
        elif test_r2 > 0.7:
            logger.info("✓ Good model performance (R² > 0.70)")
        elif test_r2 > 0.6:
            logger.info("⚠ Acceptable model performance (R² > 0.60)")
        else:
            logger.warning("⚠ Poor model performance (R² < 0.60)")

        logger.info("=" * 60)
        logger.info("Evaluation complete!")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise



if __name__ == "__main__":
    main()
