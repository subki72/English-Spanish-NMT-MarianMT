"""In-memory sliding window rate limiter for FastAPI microservices.
Provides per-IP throttling conforming to RFC 6585 with standard rate-limit headers:
- X-RateLimit-Limit
- X-RateLimit-Remaining
- X-RateLimit-Reset
- Retry-After (on HTTP 429)
"""

import asyncio
import math
import time
from collections import defaultdict

from fastapi import HTTPException, Request, status


class SlidingWindowRateLimiter:
    """Thread-safe sliding window rate limiter tracking requests per client IP."""

    def __init__(
        self,
        requests_limit: int = 60,
        window_seconds: int = 60,
        enabled: bool = True,
    ):
        self.requests_limit = requests_limit
        self.window_seconds = window_seconds
        self.enabled = enabled
        self._requests: dict[str, list[float]] = defaultdict(list)
        self._lock = asyncio.Lock()
        self._last_cleanup = time.monotonic()

    @staticmethod
    def get_client_ip(request: Request) -> str:
        """Extract client IP address respecting X-Forwarded-For and X-Real-IP proxies."""
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            # First non-empty IP in comma-separated list is the original client
            for part in forwarded.split(","):
                client = part.strip()
                if client:
                    return client
        real_ip = request.headers.get("x-real-ip")
        if real_ip and real_ip.strip():
            return real_ip.strip()
        if request.client and request.client.host:
            return request.client.host
        return "127.0.0.1"

    def _cleanup_expired(self, now: float) -> None:
        """Evict stale timestamp records to prevent memory leak over time."""
        if now - self._last_cleanup < 30.0:
            return
        self._last_cleanup = now
        cutoff = now - self.window_seconds
        expired_keys = []
        for ip, timestamps in self._requests.items():
            valid = [ts for ts in timestamps if ts > cutoff]
            if valid:
                self._requests[ip] = valid
            else:
                expired_keys.append(ip)
        for ip in expired_keys:
            del self._requests[ip]

    async def check_rate_limit(self, request: Request) -> dict[str, str]:
        """Check whether the client IP has exceeded the allowed request quota.

        Returns
        -------
        dict[str, str]
            Standard rate-limit headers to be added to the response.

        Raises
        ------
        HTTPException (429)
            If the client has exceeded requests_limit within window_seconds.
        """
        if not self.enabled:
            return {}

        client_ip = self.get_client_ip(request)
        now = time.monotonic()
        now_epoch = time.time()

        async with self._lock:
            self._cleanup_expired(now)
            cutoff = now - self.window_seconds
            valid_timestamps = [ts for ts in self._requests[client_ip] if ts > cutoff]
            self._requests[client_ip] = valid_timestamps

            if len(valid_timestamps) >= self.requests_limit:
                # Quota exceeded -> calculate wait time until oldest request leaves window
                oldest_ts = valid_timestamps[0]
                retry_after = max(1, math.ceil(self.window_seconds - (now - oldest_ts)))
                reset_epoch = int(now_epoch + retry_after)

                headers = {
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(self.requests_limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(reset_epoch),
                }
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=(
                        f"Rate limit exceeded: maximum {self.requests_limit} requests "
                        f"per {self.window_seconds}s. Try again in {retry_after} seconds."
                    ),
                    headers=headers,
                )

            # Record this request
            valid_timestamps.append(now)
            self._requests[client_ip] = valid_timestamps
            remaining = max(0, self.requests_limit - len(valid_timestamps))
            time_until_reset = max(1.0, self.window_seconds - (now - valid_timestamps[0]))
            reset_epoch = int(now_epoch + time_until_reset)

            return {
                "X-RateLimit-Limit": str(self.requests_limit),
                "X-RateLimit-Remaining": str(remaining),
                "X-RateLimit-Reset": str(reset_epoch),
            }

    def reset(self) -> None:
        """Reset state across all client IPs (useful in automated tests)."""
        self._requests.clear()
