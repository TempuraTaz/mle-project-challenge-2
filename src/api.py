# House Price Prediction API
# Simple FastAPI application for serving ML predictions

# ============================================================================
# IMPORTS & CONFIGURATION
# ============================================================================

# Standard library imports
import logging
import time

# Third-party imports
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

# Project imports
from src.config import Config
from src.model_loader import load_model_artifacts, ModelArtifacts
from src.schemas import HouseInput, MinimalHouseInput, PredictionResponse
from src.services import fill_missing_features, join_demographics_and_predict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI app instance
app = FastAPI(
    title=Config.API_TITLE,
    description=Config.API_DESCRIPTION,
    version=Config.API_VERSION
)

# Add CORS middleware to allow requests from browser (demo page)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for demo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# REQUEST LOGGING MIDDLEWARE
# ============================================================================

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Log all incoming requests with timing information.
    Helps trace what requests are coming in and how long they take.
    """
    start_time = time.time()

    logger.info(f"→ {request.method} {request.url.path}")

    response = await call_next(request)

    duration_ms = (time.time() - start_time) * 1000
    logger.info(f"← {request.method} {request.url.path} - {response.status_code} ({duration_ms:.0f}ms)")

    return response


# ============================================================================
# VALIDATION ERROR HANDLER
# ============================================================================

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Handle Pydantic validation errors with proper logging.
    This logs exactly which fields failed validation and why.
    """
    errors = exc.errors()

    logger.error(f"❌ Validation failed on {request.method} {request.url.path}")
    for error in errors:
        field = " -> ".join(str(loc) for loc in error["loc"])
        logger.error(f"   Field: {field}")
        logger.error(f"   Error: {error['msg']}")
        logger.error(f"   Type: {error['type']}")

    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation failed",
            "details": errors
        }
    )


# Global variable to store loaded model artifacts
# Will be loaded once when the server starts
artifacts: ModelArtifacts = None


# ============================================================================
# APPLICATION STARTUP
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """
    This function runs ONCE when the server starts up.
    We load all model artifacts into memory using the centralized loader.
    This is much faster than loading them for every prediction request.
    """
    global artifacts

    try:
        # Use centralized model loader
        artifacts = load_model_artifacts()
        logger.info(f"API ready to serve predictions (Model: {artifacts.model_version})")

    except Exception as e:
        logger.error(f"Failed to start API: {e}")
        raise


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint - basic info about the API"""
    return {
        "message": Config.API_TITLE,
        "version": Config.API_VERSION,
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint - tells you if the API is working"""
    return {
        "status": "healthy",
        "model_loaded": artifacts is not None and artifacts.model is not None,
        "demographics_loaded": artifacts is not None and artifacts.demographics is not None,
        "features_count": len(artifacts.model_features) if artifacts and artifacts.model_features else 0
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_house_price(house_input: HouseInput):
    """
    Main prediction endpoint.

    Send house data, get back predicted price.
    The API automatically joins demographic data on the backend.
    """
    try:
        # Convert Pydantic model to regular dictionary
        house_data = house_input.model_dump()

        # Do the prediction (this is where the magic happens)
        result = join_demographics_and_predict(house_data, artifacts)

        # Return the result (FastAPI automatically converts to JSON)
        return result

    except HTTPException:
        # Re-raise HTTP errors (like bad zipcode)
        raise
    except Exception as e:
        # Catch any other errors and return a 500 error
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict-minimal", response_model=PredictionResponse)
async def predict_minimal(house_input: MinimalHouseInput):
    """
    Flexible prediction endpoint - accepts partial data!

    Only zipcode is required. Missing features are filled with training data medians.
    This endpoint handles scenarios where only partial property data is available.

    The response includes which fields were filled with defaults for transparency.
    """
    try:
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

    except HTTPException:
        # Re-raise HTTP errors (like bad zipcode)
        raise
    except Exception as e:
        # Catch any other errors and return a 500 error
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )