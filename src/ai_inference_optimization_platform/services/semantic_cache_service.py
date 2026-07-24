from ai_inference_optimization_platform.logging.logger import logger
from ai_inference_optimization_platform.services.embedding_service import (
    EmbeddingService,
)


class SemanticCacheService:
    """
    Service responsible for semantic cache operations.
    """

    def __init__(self):

        self.embedding_service = EmbeddingService()

        logger.info("SemanticCacheService initialized.")

    async def find_similar(
        self,
        prompt: str,
    ):

        embedding = await self.embedding_service.generate_embedding(
            prompt
        )

        logger.info(
            f"Embedding length: {len(embedding)}"
        )

        return None