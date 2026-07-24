class MetricsService:
    """Service responsible for application metrics."""

    def __init__(self):
        self.cache_hits = 0
        self.cache_misses = 0

    def cache_hit(self):
        self.cache_hits += 1

    def cache_miss(self):
        self.cache_misses += 1

    def get_metrics(self):
        total = self.cache_hits + self.cache_misses
        hit_rate = 0

        if total > 0:
            hit_rate = self.cache_hits / total

        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": round(hit_rate * 100, 2),
        }


#  Sınıf tanımı bittikten sonra en sola (girintisiz) yazmalısın:
metrics_service = MetricsService()