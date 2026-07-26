from ai_inference_optimization_platform.logging.logger import logger
from ai_inference_optimization_platform.services.metrics_service import (
    metrics_service,
)
from ai_inference_optimization_platform.services.vector_store.faiss_store import (
    FAISSStore,
)


class SemanticCacheService:
    """Service responsible for semantic cache operations using FAISS vector search."""

    def __init__(self) -> None:
        self.store = FAISSStore()
        self.store.load("data/vector_store/index.faiss")

        logger.info("SemanticCacheService initialized.")

    async def find_similar(
        self,
        embedding: list[float],
        threshold: float = 0.85,
    ) -> str | None:
        # FAISS üzerinde en yakın 5 adayı ara
        result = self.store.search(embedding=embedding, top_k=5)

        if result is None:
            logger.info("Semantic Cache MISS (Vector store is empty)")
            return None

        scores, indices = result

        # En iyi adayı (index 0) değerlendir
        best_score = float(scores[0][0])
        best_index = int(indices[0][0])

        logger.info(f"FAISS Best similarity score: {best_score:.4f}")

        if best_score >= threshold and best_index != -1:
            matched_metadata = self.store.metadata[best_index]
            matched_prompt = matched_metadata.get("prompt", "")

            logger.info(
                f'Semantic Cache HIT ({best_score:.4f}) -> Matched with: "{matched_prompt}"'
            )

            # Metrics bildirimi
            metrics_service.semantic_hit()

            return matched_metadata.get("response")

        logger.info("Semantic Cache MISS")
        return None

    async def add(
        self,
        prompt: str,
        embedding: list[float],
        response: str,
    ) -> None:
        self.store.add(
            embedding=embedding,
            prompt=prompt,
            response=response,
        )

        self.store.save("data/vector_store/index.faiss")