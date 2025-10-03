"""
API Schema Definitions

Pydantic models for request/response validation in the House Price Prediction API.
"""

from typing import Optional, List
from pydantic import BaseModel, Field, ConfigDict
from src.config import Config


class HouseInput(BaseModel):
    """
    Input model for V5 model (XGBoost) - all 17 house features required.

    These are the features the V5 model was trained on:
    - 17 house features (all property characteristics)
    - Demographics are automatically joined via zipcode on the backend (26 features)

    Total model features: 43 (17 house + 26 demographics)
    """
    # Basic property info
    bedrooms: int = Field(..., ge=0, le=20, description="Number of bedrooms")
    bathrooms: float = Field(..., ge=0, le=10, description="Number of bathrooms (fractional allowed)")

    # Size features
    sqft_living: int = Field(..., ge=100, le=15000, description="Interior living space square feet")
    sqft_lot: int = Field(..., ge=0, le=1000000, description="Land lot size square feet")
    sqft_above: int = Field(..., ge=0, le=15000, description="Square feet above ground level")
    sqft_basement: int = Field(..., ge=0, le=5000, description="Basement square feet (0 if no basement)")

    # Structure features
    floors: float = Field(..., ge=1, le=5, description="Number of floors/levels")

    # Quality indicators
    waterfront: int = Field(..., ge=0, le=1, description="Waterfront property (0=No, 1=Yes)")
    view: int = Field(..., ge=0, le=4, description="View quality (0=None, 1=Fair, 2=Average, 3=Good, 4=Excellent)")
    condition: int = Field(..., ge=1, le=5, description="Overall condition (1=Poor, 3=Average, 5=Very Good)")
    grade: int = Field(..., ge=1, le=13, description="Build quality (1-3=Poor, 7=Average, 11-13=Luxury)")

    # Age features
    yr_built: int = Field(..., ge=1900, le=2025, description="Year house was built")
    yr_renovated: int = Field(..., ge=0, le=2025, description="Year renovated (0 if never)")

    # Location
    lat: float = Field(..., ge=47.0, le=48.0, description="Latitude coordinate")
    long: float = Field(..., ge=-123.0, le=-121.0, description="Longitude coordinate")
    zipcode: str = Field(..., min_length=5, max_length=5, description="5-digit zipcode")

    # Neighborhood context
    sqft_living15: int = Field(..., ge=100, le=10000, description="Avg living space of 15 nearest neighbors")
    sqft_lot15: int = Field(..., ge=0, le=1000000, description="Avg lot size of 15 nearest neighbors")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "bedrooms": 3,
                "bathrooms": 2.5,
                "sqft_living": 2000,
                "sqft_lot": 8000,
                "sqft_above": 2000,
                "sqft_basement": 0,
                "floors": 1.5,
                "waterfront": 0,
                "view": 0,
                "condition": 3,
                "grade": 7,
                "yr_built": 1990,
                "yr_renovated": 0,
                "lat": 47.5112,
                "long": -122.257,
                "zipcode": "98103",
                "sqft_living15": 1840,
                "sqft_lot15": 7620
            }
        }
    )


class MinimalHouseInput(BaseModel):
    """
    Flexible input model - accepts partial data with intelligent defaults.

    Only zipcode is required. All other 17 house fields are optional and will be filled
    with intelligent defaults (median values from training data) if not provided.

    This endpoint is designed for scenarios where only partial property data is available.
    """
    # Required
    zipcode: str = Field(..., min_length=5, max_length=5, description="5-digit zipcode (REQUIRED)")

    # Optional - Basic info
    bedrooms: Optional[int] = Field(None, ge=0, le=20, description="Number of bedrooms (optional)")
    bathrooms: Optional[float] = Field(None, ge=0, le=10, description="Number of bathrooms (optional)")

    # Optional - Size
    sqft_living: Optional[int] = Field(None, ge=100, le=15000, description="Living space square feet (optional)")
    sqft_lot: Optional[int] = Field(None, ge=0, le=1000000, description="Lot size square feet (optional)")
    sqft_above: Optional[int] = Field(None, ge=0, le=15000, description="Above ground square feet (optional)")
    sqft_basement: Optional[int] = Field(None, ge=0, le=5000, description="Basement square feet (optional)")

    # Optional - Structure
    floors: Optional[float] = Field(None, ge=1, le=5, description="Number of floors (optional)")

    # Optional - Quality
    waterfront: Optional[int] = Field(None, ge=0, le=1, description="Waterfront property (optional)")
    view: Optional[int] = Field(None, ge=0, le=4, description="View quality (optional)")
    condition: Optional[int] = Field(None, ge=1, le=5, description="Overall condition (optional)")
    grade: Optional[int] = Field(None, ge=1, le=13, description="Build quality (optional)")

    # Optional - Age
    yr_built: Optional[int] = Field(None, ge=1900, le=2025, description="Year built (optional)")
    yr_renovated: Optional[int] = Field(None, ge=0, le=2025, description="Year renovated (optional)")

    # Optional - Location
    lat: Optional[float] = Field(None, ge=47.0, le=48.0, description="Latitude (optional)")
    long: Optional[float] = Field(None, ge=-123.0, le=-121.0, description="Longitude (optional)")

    # Optional - Neighborhood
    sqft_living15: Optional[int] = Field(None, ge=100, le=10000, description="Avg neighbor living space (optional)")
    sqft_lot15: Optional[int] = Field(None, ge=0, le=1000000, description="Avg neighbor lot size (optional)")

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "description": "Full data provided",
                    "value": {
                        "zipcode": "98103",
                        "bedrooms": 3,
                        "bathrooms": 2.0,
                        "sqft_living": 1500,
                        "sqft_lot": 7000,
                        "floors": 1.0,
                        "sqft_above": 1500,
                        "sqft_basement": 0
                    }
                },
                {
                    "description": "Minimal data - only zipcode and bedrooms",
                    "value": {
                        "zipcode": "98103",
                        "bedrooms": 4
                    }
                },
                {
                    "description": "Just zipcode - maximum defaults",
                    "value": {
                        "zipcode": "98103"
                    }
                }
            ]
        }
    )


class PredictionResponse(BaseModel):
    """
    API response model - what we send back to the client
    """
    predicted_price: float = Field(..., description="Predicted house price in USD")
    zipcode: str = Field(..., description="Zipcode used for prediction")
    model_version: str = Field(..., description="Version of the model used")
    defaults_applied: Optional[List[str]] = Field(None, description="List of fields filled with defaults (only in /predict-minimal)")

    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "example": {
                "predicted_price": 485000.50,
                "zipcode": "98103",
                "model_version": Config.MODEL_VERSION,
                "defaults_applied": ["floors", "sqft_basement"]
            }
        }
    )
