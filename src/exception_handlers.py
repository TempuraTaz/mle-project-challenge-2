"""
Exception handlers for the FastAPI application.
"""
import logging
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Handle Pydantic validation errors with proper logging.
    This logs exactly which fields failed validation and why.
    """
    errors = exc.errors()

    logger.error(f"Validation failed on {request.method} {request.url.path}")
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
