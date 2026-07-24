import redis.asyncio as redis

from ai_inference_optimization_platform.config.settings import settings


class RedisClient:
    """
    Redis connection manager.
    """

    def __init__(self):

        self.client = redis.from_url(
            settings.redis_url,
            decode_responses=True,
        )

    async def ping(self):

        return await self.client.ping()

    async def get(
        self,
        key: str,
    ):

        return await self.client.get(key)

    async def set(
        self,
        key: str,
        value: str,
        ttl: int = 3600,
    ):

        await self.client.set(
            key,
            value,
            ex=ttl,
        )

    async def delete(
        self,
        key: str,
    ):

        await self.client.delete(key)