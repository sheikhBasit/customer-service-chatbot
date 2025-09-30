from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from starlette.types import ASGIApp
import logging
import json
from typing import Awaitable, Callable, Dict, Any

logger = logging.getLogger(__name__)

class LimitRequestSizeMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, max_content_length: int) -> None:
        """
        Middleware to limit request size
        
        Args:
            app: The ASGI application
            max_content_length: Maximum allowed content length in bytes
        """
        super().__init__(app)
        self.max_content_length = max_content_length

    async def dispatch(
        self, 
        request: Request, 
        call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """
        Process incoming request and enforce size limit
        
        Args:
            request: Incoming request
            call_next: Next middleware or route handler
            
        Returns:
            Response: Either error response or result from next handler
        """
        content_length = request.headers.get("Content-Length")
        error_response: Dict[str, Any]
        
        try:
            if content_length and int(content_length) > self.max_content_length:
                error_msg = f"Request too large. Maximum allowed size is {self.max_content_length} bytes."
                logger.warning(error_msg)
                error_response = {"detail": error_msg}
                return Response(
                    content=json.dumps(error_response),
                    status_code=413,
                    media_type="application/json"
                )
        except ValueError as e:
            error_msg = f"Invalid Content-Length header: {content_length}"
            logger.warning(f"{error_msg} - {str(e)}")
            error_response = {"detail": "Invalid Content-Length header."}
            return Response(
                content=json.dumps(error_response),
                status_code=400,
                media_type="application/json"
            )

        return await call_next(request)