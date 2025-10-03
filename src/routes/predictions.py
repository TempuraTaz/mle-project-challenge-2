"""
Prediction routes for house price estimation.
"""
import logging
from functools import wraps
from fastapi import APIRouter, HTTPException, Depends

from src.model_loader import ModelArtifacts
from src.schemas import HouseInput, MinimalHouseInput, PredictionResponse
from src.services import fill_missing_features, join_demographics_and_predict

logger = logging.getLogger(__name__)

router = APIRouter()

# Will be set by api.py during startup
artifacts = None


def handle_prediction_errors(func):
    """
    Decorator to handle errors in prediction endpoints.

    Catches all exceptions, logs them with full stack trace,
    and returns a clean error message to the client.
    Re-raises HTTPExceptions to preserve status codes (400, 503, etc.)
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except HTTPException:
            # Re-raise HTTP errors (like bad zipcode, model not loaded)
            raise
        except Exception as e:
            # Catch any other errors and return a 500 error
            logger.error(f"Prediction failed: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail="Prediction failed. Please check your input data."
            )
    return wrapper


def get_artifacts() -> ModelArtifacts:
    """
    Dependency that provides model artifacts to endpoints.
    Returns 503 if artifacts aren't loaded yet.
    """
    if artifacts is None:
        raise HTTPException(
            status_code=503,
            detail="Model artifacts not loaded. Server may still be starting up."
        )
    return artifacts


@router.post("/predict", response_model=PredictionResponse)
@handle_prediction_errors
async def predict_house_price(
    house_input: HouseInput,
    artifacts: ModelArtifacts = Depends(get_artifacts)
):
    """
    Main prediction endpoint.

    Send house data, get back predicted price.
    The API automatically joins demographic data on the backend.
    """
    # Convert Pydantic model to regular dictionary
    house_data = house_input.model_dump()

    # Do the prediction (this is where the magic happens)
    result = join_demographics_and_predict(house_data, artifacts)

    # Return the result (FastAPI automatically converts to JSON)
    return result


@router.post("/predict-minimal", response_model=PredictionResponse)
@handle_prediction_errors
async def predict_minimal(
    house_input: MinimalHouseInput,
    artifacts: ModelArtifacts = Depends(get_artifacts)
):
    """
    Flexible prediction endpoint - accepts partial data!

    Only zipcode is required. Missing features are filled with training data medians.
    This endpoint handles scenarios where only partial property data is available.

    The response includes which fields were filled with defaults for transparency.
    """
    # Convert Pydantic model to regular dictionary (exclude None values)
    house_data = house_input.model_dump(exclude_none=True)

    # Fill missing features with training data medians
    completed_data, defaults_applied = fill_missing_features(house_data, artifacts)

    # Make prediction with completed data
    result = join_demographics_and_predict(completed_data, artifacts)

    # Add transparency: tell user which fields were filled
    result["defaults_applied"] = defaults_applied if defaults_applied else None

    # Return the result (FastAPI automatically converts to JSON)
    return result