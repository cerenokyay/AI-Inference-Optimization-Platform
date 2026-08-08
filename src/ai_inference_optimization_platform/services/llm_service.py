import time
from datetime import datetime, timezone
from typing import AsyncGenerator

from ai_inference_optimization_platform.config.settings import settings
from ai_inference_optimization_platform.database.metrics_db import MetricsDatabase
from ai_inference_optimization_platform.logging.logger import logger
from ai_inference_optimization_platform.services.benchmark_service import (
    benchmark_service,
)
from ai_inference_optimization_platform.services.cache_service import CacheService
from ai_inference_optimization_platform.services.embedding_service import (
    EmbeddingService,
)
from ai_inference_optimization_platform.services.metrics_service import (
    metrics_service,
)
from ai_inference_optimization_platform.services.providers.provider_factory import (
    ProviderFactory,
)
from ai_inference_optimization_platform.services.semantic_cache_service import (
    SemanticCacheService,
)
from ai_inference_optimization_platform.utils.hashing import generate_prompt_hash
from ai_inference_optimization_platform.utils.prompt_normalizer import normalize_prompt
from ai_inference_optimization_platform.utils.prompt_builder import PromptBuilder


class LLMService:
    """Service responsible for interacting with language model providers and streaming."""

    def __init__(self) -> None:
        self.provider = ProviderFactory.create()
        self.cache = CacheService()
        self.semantic_cache = SemanticCacheService()
        self.embedding_service = EmbeddingService()

        # Sağlayıcı ve model isimlerini metrikler için baştan belirliyoruz (Varsayılanlar)
        self.provider_name = self.provider.__class__.__name__
        self.model_name = getattr(self.provider, "model", settings.default_model)

        logger.info("LLMService initialized.")

    # ✨ YENİ: Dışarıdan dinamik model ve sağlayıcı seçimi eklendi
    async def generate_stream(
        self, 
        prompt: str,
        provider_override: str = None,
        model_override: str = None,
        api_key: str = None
    ) -> AsyncGenerator[str, None]:
        benchmark_service.record_request()
        request_start = time.perf_counter()

        logger.info("Generating streaming response.")

        normalized_prompt = normalize_prompt(prompt)
        prompt_hash = generate_prompt_hash(normalized_prompt)

        # Dinamik Sağlayıcı (Provider) Belirleme
        active_provider = self.provider
        current_provider_name = self.provider_name
        current_model_name = self.model_name

        if provider_override and model_override:
            active_provider = ProviderFactory.create(
                provider_type=provider_override, 
                model=model_override, 
                api_key=api_key
            )
            current_provider_name = active_provider.__class__.__name__
            current_model_name = getattr(active_provider, "model", model_override)

        embedding_start = time.perf_counter()
        embedding = await self.embedding_service.generate_embedding(normalized_prompt)
        benchmark_service.record_embedding_time(
            (time.perf_counter() - embedding_start) * 1000
        )

        # ==========================================
        # 1. Semantic Cache Kontrolü
        # ==========================================
        semantic_start = time.perf_counter()
        semantic_response = await self.semantic_cache.find_similar(embedding=embedding)
        benchmark_service.record_semantic_search_time(
            (time.perf_counter() - semantic_start) * 1000
        )

        if semantic_response is not None:
            logger.info("Returning semantic cached response as stream.")
            yield semantic_response
            
            total_latency = (time.perf_counter() - request_start) * 1000
            benchmark_service.record_request_time(total_latency)
            
            # Veritabanına Logla
            await MetricsDatabase.save_metric(
                timestamp=datetime.now(timezone.utc).isoformat(),
                prompt_hash=prompt_hash,
                provider=current_provider_name,
                model_name=current_model_name,
                cache_status="SEMANTIC_HIT",
                total_latency_ms=total_latency,
            )
            return

        # ==========================================
        # 2. Exact Cache Kontrolü (Redis)
        # ==========================================
        cache_start = time.perf_counter()
        cached_response = await self.cache.get(prompt_hash)
        benchmark_service.record_cache_lookup_time(
            (time.perf_counter() - cache_start) * 1000
        )

        if cached_response is not None:
            logger.info("Returning exact cached response as stream.")
            await self.semantic_cache.add(
                prompt=normalized_prompt,
                embedding=embedding,
                response=cached_response,
            )
            yield cached_response
            
            total_latency = (time.perf_counter() - request_start) * 1000
            benchmark_service.record_request_time(total_latency)
            
            # Veritabanına Logla
            await MetricsDatabase.save_metric(
                timestamp=datetime.now(timezone.utc).isoformat(),
                prompt_hash=prompt_hash,
                provider=current_provider_name,
                model_name=current_model_name,
                cache_status="EXACT_HIT",
                total_latency_ms=total_latency,
            )
            return

        # ==========================================
        # 3. Provider Çağrısı (LLM Fallback - Streaming)
        # ==========================================
        logger.info(f"Streaming response from provider: {current_provider_name} ({current_model_name})")
        metrics_service.provider_call()

        provider_start = time.perf_counter()
        full_response_chunks = []

        # Test vs Mimari Sohbet Ayrımı
        if "[test_mode]" in normalized_prompt:
            enriched_prompt = normalized_prompt.replace("[test_mode]", "").strip()
        else:
            enriched_prompt = PromptBuilder.build_final_prompt(normalized_prompt)

        # ✨ YENİ: Güvenli Streaming Çağrısı (Try-Except ile)
        try:
            async for chunk in active_provider.generate_stream(enriched_prompt):
                full_response_chunks.append(chunk)
                yield chunk
        except Exception as e:
            logger.error(f"Provider Error: {e}")
            yield f"\n\n**[SİSTEM HATASI]**: Seçili model ({current_model_name}) ile iletişim kurulamadı. Lütfen model ayarlarınızı veya API anahtarınızı kontrol edin. Detay: {str(e)}"
            return

        benchmark_service.record_provider_time(
            (time.perf_counter() - provider_start) * 1000
        )

        full_response = "".join(full_response_chunks)

        # Üretim Bittiğinde Cache'lere Kaydet
        await self.cache.set(key=prompt_hash, value=full_response)
        await self.semantic_cache.add(
            prompt=normalized_prompt,
            embedding=embedding,
            response=full_response,
        )

        total_latency = (time.perf_counter() - request_start) * 1000
        benchmark_service.record_request_time(total_latency)

        # Veritabanına Logla
        await MetricsDatabase.save_metric(
            timestamp=datetime.now(timezone.utc).isoformat(),
            prompt_hash=prompt_hash,
            provider=current_provider_name,
            model_name=current_model_name,
            cache_status="MISS",
            total_latency_ms=total_latency,
        )