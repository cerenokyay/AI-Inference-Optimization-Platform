import time

from ai_inference_optimization_platform.logging.logger import logger
from ai_inference_optimization_platform.services.benchmark_service import (
    benchmark_service,
)
from ai_inference_optimization_platform.services.embeddings.embedding_factory import (
    EmbeddingFactory,
)


class EmbeddingService:
    """Service responsible for embedding generation and latency benchmarking."""

    def __init__(self) -> None:
        self.provider = EmbeddingFactory.create()
        logger.info("EmbeddingService initialized.")

    async def generate_embedding(
        self,
        text: str,
    ) -> list[float]:
        start = time.perf_counter()

        embedding = await self.provider.generate_embedding(text)

        elapsed_ms = (time.perf_counter() - start) * 1000

        # Benchmark servisine süreyi kaydet
        benchmark_service.record_embedding_time(elapsed_ms)

        logger.info(f"Embedding generated in {elapsed_ms:.2f} ms")

        return embedding