"""Correlation ID & Structured Logging Middleware.
Ensures every request has an X-Request-ID and logs structured JSON containing:
- request_id
- client_ip
- latency_ms
- tokens_generated
- status_code
"""

import json
import logging
import time
import uuid
from datetime import UTC, datetime
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from app.config import settings
from app.rate_limiter import SlidingWindowRateLimiter

structured_logger = logging.getLogger("nmt.structured")


class RequestIdAndStructuredLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for correlation IDs (X-Request-ID) and structured JSON audit logging."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # 1. Resolve or generate Request ID
        request_id = request.headers.get("x-request-id") or request.headers.get("X-Request-ID")
        if not request_id or not request_id.strip():
            request_id = str(uuid.uuid4())
        else:
            request_id = request_id.strip()

        request.state.request_id = request_id

        # 2. Extract Client IP
        client_ip = SlidingWindowRateLimiter.get_client_ip(request)
        request.state.client_ip = client_ip

        # 3. Initialize metrics container on request.state
        request.state.tokens_generated = 0

        start_time = time.perf_counter()
        status_code = 500
        error_msg: str | None = None

        try:
            response = await call_next(request)
            status_code = response.status_code

            # Always inject X-Request-ID into downstream response headers
            response.headers["X-Request-ID"] = request_id
            return response

        except Exception as exc:
            status_code = 500
            error_msg = str(exc)
            raise

        finally:
            if settings.enable_structured_logging:
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                tokens_generated = getattr(request.state, "tokens_generated", 0)

                log_entry: dict[str, Any] = {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "request_id": request_id,
                    "client_ip": client_ip,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": status_code,
                    "latency_ms": round(latency_ms, 2),
                    "tokens_generated": tokens_generated,
                }
                if error_msg is None and status_code >= 500:
                    state_error = getattr(request.state, "error_message", None)
                    if state_error:
                        error_msg = str(state_error)

                if error_msg is not None:
                    log_entry["error"] = error_msg

                # Log level according to status code
                if status_code >= 500:
                    structured_logger.error(json.dumps(log_entry))
                elif status_code >= 400:
                    structured_logger.warning(json.dumps(log_entry))
                else:
                    structured_logger.info(json.dumps(log_entry))
