from ai_inference_optimization_platform.logging.logger import logger
from ai_inference_optimization_platform.services.providers.embedding_provider import (
    EmbeddingProvider,
)


class OllamaEmbeddingProvider(EmbeddingProvider):

    async def embed(
        self,
        text: str,
    ) -> list[float]:

        logger.info("Generating embedding.")

        return []