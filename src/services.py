"""
Business Logic Services

Core prediction and data processing logic for the House Price Prediction API.
Separated from API endpoints for better testability and maintainability.
"""

from typing import Tuple, List
import pandas as pd
from fastapi import HTTPException

from src.model_loader import ModelArtifacts


def fill_missing_features(house_data: dict, artifacts: ModelArtifacts) -> Tuple[dict, List[str]]:
    """
    Fill missing house features with training data medians.

    Strategy: Simple and transparent - use median values from training data.
    Medians are robust to outliers and represent typical Seattle homes.

    Note: For production, data science team could implement more sophisticated
    imputation methods (KNN imputer, iterative imputer, etc.)

    Args:
        house_data: Dictionary with partial house features (zipcode always present)
        artifacts: Loaded model artifacts containing feature defaults

    Returns:
        Tuple of (completed_house_data, list_of_defaults_applied)
    """
    completed = house_data.copy()
    defaults_applied = []

    # Fill each missing feature with its training data median
    for feature, default_value in artifacts.feature_defaults.items():
        if completed.get(feature) is None:
            completed[feature] = default_value
            defaults_applied.append(feature)

    return completed, defaults_applied


def join_demographics_and_predict(house_data: dict, artifacts: ModelArtifacts) -> dict:
    """
    Core prediction logic:
    1. Take house data from the client
    2. Look up demographics for that zipcode
    3. Combine house + demographic features in the correct order
    4. Make prediction using the loaded model
    5. Return structured result

    Args:
        house_data: Dictionary with house features (including zipcode)
        artifacts: Loaded model artifacts (model, demographics, features, etc.)

    Returns:
        Dictionary with predicted_price, zipcode, and model_version

    Raises:
        HTTPException: If zipcode not found in demographics data
    """
    zipcode = house_data["zipcode"]

    # Look up demographics for this zipcode
    demo_row = artifacts.demographics[artifacts.demographics["zipcode"] == zipcode]

    if demo_row.empty:
        # This zipcode isn't in our demographics data
        available_zipcodes = sorted(artifacts.demographics["zipcode"].unique())[:5]  # First 5 for error message
        raise HTTPException(
            status_code=400,
            detail=f"Zipcode '{zipcode}' not found in demographics. Available: {available_zipcodes}..."
        )

    # Build feature array in the exact order the model expects
    feature_values = []
    for feature_name in artifacts.model_features:
        if feature_name in house_data:
            # This is a house feature (came from client)
            feature_values.append(house_data[feature_name])
        else:
            # This must be a demographic feature (from our lookup table)
            demographic_value = demo_row[feature_name].iloc[0]
            feature_values.append(demographic_value)

    # Convert to DataFrame to avoid sklearn feature name warnings
    X = pd.DataFrame([feature_values], columns=artifacts.model_features)

    # Make prediction using the loaded model
    prediction = artifacts.model.predict(X)[0]

    # Return structured result
    return {
        "predicted_price": float(prediction),  # Convert numpy float to Python float
        "zipcode": zipcode,
        "model_version": artifacts.model_version
    }
