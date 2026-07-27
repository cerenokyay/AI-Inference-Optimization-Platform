import time

from ai_inference_optimization_platform.cache.redis_client import RedisClient
from ai_inference_optimization_platform.config.settings import settings
from ai_inference_optimization_platform.logging.logger import logger
from ai_inference_optimization_platform.services.benchmark_service import (
    benchmark_service,
)
from ai_inference_optimization_platform.services.metrics_service import (
    metrics_service,
)


class CacheService:
    """Service responsible for cache operations and Redis latency benchmarking."""

    def __init__(self) -> None:
        self.redis = RedisClient()

    async def get(self, key: str) -> str | None:
        logger.info(f"Checking cache for key: {key}")

        start = time.perf_counter()
        value = await self.redis.get(key)
        elapsed_ms = (time.perf_counter() - start) * 1000

        benchmark_service.record_cache_lookup_time(elapsed_ms)
        logger.info(f"Redis lookup took {elapsed_ms:.2f} ms")

        if value is None:
            logger.info("Cache MISS")
            metrics_service.cache_miss()
        else:
            logger.info("Cache HIT")
            metrics_service.cache_hit()

        return value

    async def set(
        self,
        key: str,
        value: str,
        ttl: int | None = None,
    ) -> None:
        # Dinamik TTL kontrolü
        if ttl is None:
            ttl = settings.cache_ttl

        logger.info(f"Saving response to cache: {key}")

        start = time.perf_counter()
        await self.redis.set(
            key=key,
            value=value,
            ttl=ttl,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.info(f"Redis SET took {elapsed_ms:.2f} ms")