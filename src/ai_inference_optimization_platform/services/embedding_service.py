from ai_inference_optimization_platform.logging.logger import logger
from ai_inference_optimization_platform.services.embeddings.embedding_factory import (
    EmbeddingFactory,
)


class EmbeddingService:
    """
    Service responsible for embedding generation.
    """

    def __init__(self):

        self.provider = EmbeddingFactory.create()

        logger.info("EmbeddingService initialized.")

    async def generate_embedding(
        self,
        text: str,
    ) -> list[float]:

        return await self.provider.generate_embedding(text)