from ai_inference_optimization_platform.logging.logger import logger


class MetricsService:
    """
    Service responsible for collecting application metrics.
    """

    def __init__(self) -> None:
        self.cache_hits = 0
        self.cache_misses = 0
        self.semantic_hits = 0
        self.provider_calls = 0

        logger.info("MetricsService initialized.")

    # ===== Eski API =====

    def cache_hit(self) -> None:
        self.cache_hits += 1
        logger.info(f"Cache Hits: {self.cache_hits}")

    def cache_miss(self) -> None:
        self.cache_misses += 1
        logger.info(f"Cache Misses: {self.cache_misses}")

    def semantic_hit(self) -> None:
        self.semantic_hits += 1
        logger.info(f"Semantic Cache Hits: {self.semantic_hits}")

    def provider_call(self) -> None:
        self.provider_calls += 1
        logger.info(f"Provider Calls: {self.provider_calls}")

    # ===== Yeni API (İstersen kalsın) =====

    def record_cache_hit(self) -> None:
        self.cache_hit()

    def record_cache_miss(self) -> None:
        self.cache_miss()

    def record_semantic_hit(self) -> None:
        self.semantic_hit()

    def record_provider_call(self) -> None:
        self.provider_call()

    def get_metrics(self) -> dict:
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "semantic_hits": self.semantic_hits,
            "provider_calls": self.provider_calls,
        }


metrics_service = MetricsService()