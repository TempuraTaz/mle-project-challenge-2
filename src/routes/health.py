"""
Health check routes for monitoring API status.
"""
from fastapi import APIRouter
from src.config import Config

router = APIRouter()

# Will be set by api.py during startup
artifacts = None


@router.get("/")
async def root():
    """Root endpoint - basic info about the API"""
    return {
        "message": Config.API_TITLE,
        "version": Config.API_VERSION,
        "docs": "/docs",
        "health": "/health"
    }


@router.get("/health")
async def health_check():
    """
    Health check endpoint - always returns 200 and reports current status.

    Returns 'starting' if model artifacts are still loading.
    Returns 'healthy' once model is fully loaded and ready.
    """
    if artifacts is None:
        return {
            "status": "starting",
            "model_loaded": False,
            "demographics_loaded": False,
            "features_count": 0
        }

    return {
        "status": "healthy",
        "model_loaded": artifacts.model is not None,
        "demographics_loaded": artifacts.demographics is not None,
        "features_count": len(artifacts.model_features) if artifacts.model_features else 0
    }