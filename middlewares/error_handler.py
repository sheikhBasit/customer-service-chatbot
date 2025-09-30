from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
import logging
from typing import Awaitable, Callable

logger = logging.getLogger(__name__)

class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, 
        request: Request, 
        call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """
        Middleware to catch and handle unhandled exceptions
        
        Args:
            request: The incoming request
            call_next: The next middleware or route handler to call
            
        Returns:
            Response: Either the normal response or error response
        """
        try:
            response = await call_next(request)
            return response
        except Exception as exc:
            logger.exception("Unhandled error occurred", exc_info=exc)
            return JSONResponse(
                status_code=500,
                content={
                    "detail": "Internal Server Error",
                    "error": str(exc)
                }
            )