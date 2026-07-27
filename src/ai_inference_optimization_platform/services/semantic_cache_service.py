import time

from ai_inference_optimization_platform.config.settings import settings
from ai_inference_optimization_platform.logging.logger import logger
from ai_inference_optimization_platform.services.benchmark_service import (
    benchmark_service,
)
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
        threshold: float | None = None,
    ) -> str | None:
        # Eğer dışarıdan threshold verilmezse settings'deki değeri kullan
        if threshold is None:
            threshold = settings.semantic_threshold

        start = time.perf_counter()

        result = self.store.search(
            embedding=embedding,
            top_k=5,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000
        benchmark_service.record_semantic_search_time(elapsed_ms)

        logger.info(f"FAISS search took {elapsed_ms:.2f} ms")

        if result is None:
            benchmark_service.record_semantic_miss()
            logger.info("Semantic Cache MISS (Vector store is empty)")
            return None

        scores, indices = result

        best_score = float(scores[0][0])
        best_index = int(indices[0][0])

        logger.info(f"FAISS Best similarity score: {best_score:.4f}")

        if best_score >= threshold and best_index != -1:
            matched_metadata = self.store.metadata[best_index]
            matched_prompt = matched_metadata.get("prompt", "")

            logger.info(
                f'Semantic Cache HIT ({best_score:.4f}) -> Matched with: "{matched_prompt}"'
            )

            benchmark_service.record_semantic_hit()
            metrics_service.semantic_hit()

            return matched_metadata.get("response")

        benchmark_service.record_semantic_miss()
        logger.info("Semantic Cache MISS")

        return None

    async def add(
        self,
        prompt: str,
        embedding: list[float],
        response: str,
    ) -> None:
        result = self.store.search(
            embedding=embedding,
            top_k=1,
        )

        if result is not None:
            scores, _ = result
            best_score = float(scores[0][0])

            # Sabit 0.99 yerine settings üzerinden gelen değeri kullan
            if best_score >= settings.duplicate_threshold:
                logger.info(
                    f"Duplicate embedding detected (score: {best_score:.4f}). Skipping insert."
                )
                return

        self.store.add(
            embedding=embedding,
            prompt=prompt,
            response=response,
        )

        self.store.save("data/vector_store/index.faiss")