from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
import logging
import time
from typing import Callable, Awaitable

logger = logging.getLogger(__name__)

class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, 
        request: Request, 
        call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """
        Log request details including method, path, status code and duration
        
        Args:
            request: Incoming request
            call_next: Next middleware or route handler
            
        Returns:
            Response: The response from the application
        """
        start_time = time.time()
        try:
            response = await call_next(request)
        except Exception as e:
            logger.error(f"Request failed: {request.method} {request.url.path} - {str(e)}")
            raise
        
        duration = time.time() - start_time
        logger.info(
            f"{request.method} {request.url.path} "
            f"status={response.status_code} "
            f"duration={duration:.2f}s"
        )
        return response