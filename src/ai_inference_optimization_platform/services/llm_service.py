from ai_inference_optimization_platform.logging.logger import logger
from ai_inference_optimization_platform.services.cache_service import CacheService
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

        logger.info("LLMService initialized.")

    async def generate(self, prompt: str) -> str:
        logger.info("Generating response.")

        # 1. Prompt'u normalize et
        normalized_prompt = normalize_prompt(prompt)
        logger.info(f"Normalized prompt: {normalized_prompt}")

        # 2. Normalize edilmiş prompt üzerinden hash üret
        prompt_hash = generate_prompt_hash(normalized_prompt)
        logger.info(f"Prompt hash: {prompt_hash}")

        # 3. Semantic Cache kontrolü yap (Benzer soru daha önce soruldu mu?)
        semantic_response = await self.semantic_cache.find_similar(
            normalized_prompt
        )

        if semantic_response is not None:
            logger.info("Semantic cache HIT.")
            return semantic_response

        # 4. Exact Match (Exact Hash) Cache kontrolü yap
        cached_response = await self.cache.get(prompt_hash)

        if cached_response is not None:
            logger.info("Returning cached response.")
            return cached_response

        # 5. Cache MISS ise gerçek LLM provider'a git
        response = await self.provider.generate(normalized_prompt)

        # 6. Yeni cevabı hem exact cache'e hem de semantic cache'e kaydet
        await self.cache.set(
            key=prompt_hash,
            value=response,
        )

        # İsteğe bağlı: Eğer semantic_cache üzerinde kaydetme metodun varsa buraya ekleyebilirsin:
        # await self.semantic_cache.add(normalized_prompt, response)

        return response