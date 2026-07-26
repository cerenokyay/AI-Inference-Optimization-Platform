from ai_inference_optimization_platform.logging.logger import logger
from ai_inference_optimization_platform.services.cache_service import CacheService
from ai_inference_optimization_platform.services.embedding_service import (
    EmbeddingService,
)
from ai_inference_optimization_platform.services.metrics_service import (
    metrics_service,
)
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

        # 1. Normalize Prompt
        normalized_prompt = normalize_prompt(prompt)
        logger.info(f"Normalized prompt: {normalized_prompt}")

        # 2. Generate Hash
        prompt_hash = generate_prompt_hash(normalized_prompt)
        logger.info(f"Prompt hash: {prompt_hash}")

        # 3. Generate Embedding
        embedding = await self.embedding_service.generate_embedding(
            normalized_prompt
        )

        # 4. Semantic Cache Check (FAISS top 5)
        semantic_response = await self.semantic_cache.find_similar(
            embedding=embedding,
            threshold=0.85,
        )

        if semantic_response is not None:
            logger.info("Returning semantic cached response.")
            return semantic_response

        # 5. Exact Cache Check (Redis)
        cached_response = await self.cache.get(prompt_hash)

        if cached_response is not None:
            logger.info("Returning exact cached response.")

            # Exact hit olan isteği vektör mağazasına da besle
            await self.semantic_cache.add(
                prompt=normalized_prompt,
                embedding=embedding,
                response=cached_response,
            )

            return cached_response

        # 6. Provider Call (LLM Fallback)
        logger.info("Generating response from provider.")

        # Metrik bildirimi: Provider çağrısı yapılıyor
        metrics_service.provider_call()

        response = await self.provider.generate(normalized_prompt)

        # 7. Exact Cache Save (Redis)
        await self.cache.set(
            key=prompt_hash,
            value=response,
        )

        # 8. Semantic Cache Save (FAISS + metadata.json)
        await self.semantic_cache.add(
            prompt=normalized_prompt,
            embedding=embedding,
            response=response,
        )

        return response