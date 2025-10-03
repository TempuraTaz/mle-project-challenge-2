# House Price Prediction API
# Simple FastAPI application for serving ML predictions

# ============================================================================
# IMPORTS & CONFIGURATION
# ============================================================================

# Standard library imports
import logging

# Third-party imports
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError

# Project imports
from src.config import Config
from src.model_loader import load_model_artifacts, ModelArtifacts
from src.middleware import log_requests
from src.exception_handlers import validation_exception_handler
from src.routes import health, predictions

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

# Add request logging middleware
app.middleware("http")(log_requests)

# Register exception handler
app.add_exception_handler(RequestValidationError, validation_exception_handler)


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
    try:
        # Use centralized model loader
        artifacts = load_model_artifacts()

        # Share artifacts with route modules
        health.artifacts = artifacts
        predictions.artifacts = artifacts

        logger.info(f"API ready to serve predictions (Model: {artifacts.model_version})")

    except Exception as e:
        logger.error(f"Failed to start API: {e}")
        raise


# ============================================================================
# REGISTER ROUTERS
# ============================================================================

# Register health check routes (/, /health)
app.include_router(health.router, tags=["Health"])

# Register prediction routes (/predict, /predict-minimal)
app.include_router(predictions.router, tags=["Predictions"])