from ai_inference_optimization_platform.cache.redis_client import RedisClient
from ai_inference_optimization_platform.logging.logger import logger
from ai_inference_optimization_platform.services.metrics_service import metrics_service


class CacheService:
    """
    Service responsible for cache operations.
    """

    def __init__(self) -> None:
        self.redis = RedisClient()
        

    async def get(self, key: str):

        logger.info(f"Checking cache for key: {key}")

        value = await self.redis.get(key)

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
        ttl: int = 3600,
    ):

        logger.info(f"Saving response to cache: {key}")

        await self.redis.set(
            key=key,
            value=value,
            ttl=ttl,
        )