"""
Request logging middleware for the FastAPI application.
"""
import logging
import time
from fastapi import Request

logger = logging.getLogger(__name__)


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
