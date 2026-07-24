from ai_inference_optimization_platform.logging.logger import logger
from ai_inference_optimization_platform.services.cache_service import CacheService
from ai_inference_optimization_platform.services.providers.ollama_provider import (
    OllamaProvider,
)
from ai_inference_optimization_platform.utils.hashing import generate_prompt_hash
from ai_inference_optimization_platform.utils.prompt_normalizer import (
    normalize_prompt,
)


class LLMService:
    """Service responsible for interacting with language model providers."""

    def __init__(self) -> None:
        self.provider = OllamaProvider()
        self.cache = CacheService()

        logger.info("LLMService initialized.")

    async def generate(self, prompt: str) -> str:
        logger.info("Generating response.")

        # 1. Prompt'u normalize et
        normalized_prompt = normalize_prompt(prompt)
        logger.info(f"Normalized prompt: {normalized_prompt}")

        # 2. Normalize edilmiş prompt üzerinden hash üret
        prompt_hash = generate_prompt_hash(normalized_prompt)
        logger.info(f"Prompt hash: {prompt_hash}")

        # 3. Cache kontrolü yap
        cached_response = await self.cache.get(prompt_hash)

        if cached_response is not None:
            logger.info("Returning cached response.")
            return cached_response

        # 4. Cache MISS ise gerçek LLM provider'a git (Ollama)
        response = await self.provider.generate(
            normalized_prompt
        ) 

        # 5. Yeni cevabı cache'e kaydet
        await self.cache.set(
            key=prompt_hash,
            value=response,
        )

        return response