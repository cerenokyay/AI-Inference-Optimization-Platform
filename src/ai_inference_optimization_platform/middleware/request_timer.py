import time

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from ai_inference_optimization_platform.logging.logger import logger


class RequestTimerMiddleware(BaseHTTPMiddleware):
    """
    Middleware that measures request processing time.
    """

    async def dispatch(self, request: Request, call_next):
        start_time = time.perf_counter()

        response = await call_next(request)

        duration = (time.perf_counter() - start_time) * 1000

        logger.info(
            f"{request.method} {request.url.path} completed in {duration:.2f} ms"
        )

        response.headers["X-Process-Time"] = f"{duration:.2f}"

        return response