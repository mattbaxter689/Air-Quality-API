from starlette.middleware.base import (
    BaseHTTPMiddleware,
    RequestResponseEndpoint,
)
from starlette.types import ASGIApp
from redis.asyncio import Redis
from fastapi import Request, status, Response
from fastapi.responses import JSONResponse
import time
import logging

logger = logging.getLogger("weather_api")


class TokenBucketMiddleware(BaseHTTPMiddleware):

    def __init__(
        self,
        app: ASGIApp,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        bucket_size: int = 10,
        refill_rate: int = 1,
    ) -> None:

        super().__init__(app)
        self.redis = Redis(
            host=redis_host, port=redis_port, decode_responses=True
        )
        self.bucket_size = bucket_size
        self.refill_rate = refill_rate

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> JSONResponse | Response:
        client_ip = request.client.host
        key = f"token_bucket:{client_ip}"
        now = int(time.time())

        data = await self.redis.hgetall(key)
        token = float(data.get("tokens", self.bucket_size))
        last_refill = int(data.get("last_refill", now))

        elapsed = max(0, now - last_refill)
        tokens = min(self.bucket_size, token + elapsed * self.refill_rate)

        if tokens < 1:
            logger.info(
                f"Too many requests sent by: {client_ip}. Limiting request rate"
            )
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"detail": "Exceeded request rate-limit"},
            )

        await self.redis.hset(
            name=key, mapping={"tokens": tokens - 1, "last_refill": now}
        )

        await self.redis.expire(key, 3600)

        return await call_next(request)
