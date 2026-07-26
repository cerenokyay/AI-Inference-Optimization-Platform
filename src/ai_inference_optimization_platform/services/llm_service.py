from ai_inference_optimization_platform.logging.logger import logger
from ai_inference_optimization_platform.services.cache_service import CacheService
from ai_inference_optimization_platform.services.embedding_service import EmbeddingService
from ai_inference_optimization_platform.services.providers.provider_factory import (
    ProviderFactory,
)
from ai_inference_optimization_platform.services.semantic_cache_service import (
    SemanticCacheService,
)
from ai_inference_optimization_platform.utils.hashing import generate_prompt_hash
from ai_inference_optimization_platform.utils.prompt_normalizer import (
    normalize_prompt,
)


class LLMService:
    """Service responsible for interacting with language model providers."""

    def __init__(self) -> None:
        self.provider = ProviderFactory.create()
        self.cache = CacheService()
        self.semantic_cache = SemanticCacheService()
        self.embedding_service = EmbeddingService()

        logger.info("LLMService initialized.")

    async def generate(self, prompt: str) -> str:
        logger.info("Generating response.")

        normalized_prompt = normalize_prompt(prompt)
        logger.info(f"Normalized prompt: {normalized_prompt}")

        prompt_hash = generate_prompt_hash(normalized_prompt)
        logger.info(f"Prompt hash: {prompt_hash}")

        embedding = await self.embedding_service.generate_embedding(
            normalized_prompt
        )

        semantic_response = await self.semantic_cache.find_similar(
            embedding=embedding,
        )

        if semantic_response is not None:
            logger.info("Returning semantic cached response.")
            return semantic_response

        cached_response = await self.cache.get(prompt_hash)

        if cached_response is not None:
            logger.info("Returning exact cached response.")

            await self.semantic_cache.add(
                prompt=normalized_prompt,
                embedding=embedding,
                response=cached_response,
            )

            return cached_response

        logger.info("Generating response from provider.")

        response = await self.provider.generate(
            normalized_prompt
        )

        await self.cache.set(
            key=prompt_hash,
            value=response,
        )

        await self.semantic_cache.add(
            prompt=normalized_prompt,
            embedding=embedding,
            response=response,
        )

        return response