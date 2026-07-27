from ai_inference_optimization_platform.logging.logger import logger


class BenchmarkService:
    """Collects latency and performance statistics including semantic cache hit rates."""

    def __init__(self) -> None:
        self.total_requests = 0

        # Latency (Gecikme) Sayaçları
        self.total_request_time = 0.0
        self.total_embedding_time = 0.0
        self.total_provider_time = 0.0
        self.total_semantic_search_time = 0.0
        self.total_cache_lookup_time = 0.0

        # Semantic Cache HIT / MISS Sayaçları
        self.semantic_hits = 0
        self.semantic_misses = 0

        logger.info("BenchmarkService initialized.")

    def record_request(self) -> None:
        self.total_requests += 1

    def record_request_time(self, elapsed_ms: float) -> None:
        self.total_request_time += elapsed_ms

    def record_embedding_time(self, elapsed_ms: float) -> None:
        self.total_embedding_time += elapsed_ms

    def record_provider_time(self, elapsed_ms: float) -> None:
        self.total_provider_time += elapsed_ms

    def record_semantic_search_time(self, elapsed_ms: float) -> None:
        self.total_semantic_search_time += elapsed_ms

    def record_cache_lookup_time(self, elapsed_ms: float) -> None:
        self.total_cache_lookup_time += elapsed_ms

    def record_semantic_hit(self) -> None:
        """Increment semantic cache hit count."""
        self.semantic_hits += 1

    def record_semantic_miss(self) -> None:
        """Increment semantic cache miss count."""
        self.semantic_misses += 1

    def get_metrics(self) -> dict:
        requests = max(self.total_requests, 1)

        # Semantic Cache Hit Rate Hesaplaması
        semantic_total = self.semantic_hits + self.semantic_misses
        semantic_hit_rate = (
            round(self.semantic_hits / semantic_total * 100, 2)
            if semantic_total > 0
            else 0.0
        )

        return {
            "total_requests": self.total_requests,
            "avg_request_ms": round(
                self.total_request_time / requests,
                2,
            ),
            "avg_embedding_ms": round(
                self.total_embedding_time / requests,
                2,
            ),
            "avg_provider_ms": round(
                self.total_provider_time / requests,
                2,
            ),
            "avg_semantic_search_ms": round(
                self.total_semantic_search_time / requests,
                2,
            ),
            "avg_cache_lookup_ms": round(
                self.total_cache_lookup_time / requests,
                2,
            ),
            # Semantic Cache Metrikleri
            "semantic_hits": self.semantic_hits,
            "semantic_misses": self.semantic_misses,
            "semantic_hit_rate": semantic_hit_rate,
        }


benchmark_service = BenchmarkService()