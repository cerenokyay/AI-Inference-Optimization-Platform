from ai_inference_optimization_platform.logging.logger import logger
from ai_inference_optimization_platform.services.embedding_service import (
    EmbeddingService,
)
from ai_inference_optimization_platform.utils.vector import cosine_similarity


class SemanticCacheService:
    """Service responsible for semantic cache operations using vector similarity."""

    def __init__(self) -> None:
        self.embedding_service = EmbeddingService()
        self.entries = []

        logger.info("SemanticCacheService initialized.")

    async def find_similar(self, embedding: list[float]):
        logger.info(f"Embedding length: {len(embedding)}")


        best_score = 0.0
        best_response = None

        for item in self.entries:
            score = cosine_similarity(
                embedding,
                item["embedding"],
            )

            logger.info(f"Similarity score: {score:.4f}")
            logger.info(f'Comparing with: "{item["prompt"]}"')

            if score > best_score:
                best_score = score
                best_response = item["response"]

        logger.info(f"Best similarity: {best_score:.4f}")


        if best_score >= 0.95:
            logger.info("Semantic Cache HIT")
            return best_response

        logger.info("Semantic Cache MISS")
        return None

    async def add(
        self,
        embedding: list[float],
        response: str,
        prompt: str,

    ) -> None:
        self.entries.append(
            {
                "prompt": prompt,
                "embedding": embedding,
                "response": response,
            }
        )

        logger.info(
            f"Stored semantic embedding. Total entries: {len(self.entries)}"
        )