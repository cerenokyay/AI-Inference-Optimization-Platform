from ai_inference_optimization_platform.logging.logger import logger
from ai_inference_optimization_platform.services.providers.ollama_provider import OllamaProvider
from ai_inference_optimization_platform.services.cache_service import CacheService
from ai_inference_optimization_platform.utils.hashing import generate_prompt_hash


class LLMService:
    """
    Service responsible for interacting with language model providers.
    """

    def __init__(self) -> None:
        self.provider = OllamaProvider()
        self.cache = CacheService()

        logger.info("LLMService initialized.")

    async def generate(self, prompt: str) -> str:

        logger.info("Generating response through provider.")

        prompt_hash = generate_prompt_hash(prompt)

        cached_response = await self.cache.get(prompt_hash)

        if cached_response is not None:
            logger.info("Returning cached response.")
            return cached_response

        response = await self.provider.generate(prompt)

        await self.cache.set(
            key=prompt_hash,
            value=response,
        )

        return response