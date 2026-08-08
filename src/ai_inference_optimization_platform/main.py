from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from ai_inference_optimization_platform.cache.redis_client import RedisClient
from ai_inference_optimization_platform.config.settings import settings
from ai_inference_optimization_platform.database.metrics_db import MetricsDatabase
from ai_inference_optimization_platform.exceptions.custom_exceptions import (
    AIInferenceException,
)
from ai_inference_optimization_platform.exceptions.handlers import (
    register_exception_handlers,
)
from ai_inference_optimization_platform.logging.logger import logger
from ai_inference_optimization_platform.middleware.request_timer import (
    RequestTimerMiddleware,
)
from ai_inference_optimization_platform.schemas.requests import GenerateRequest
from ai_inference_optimization_platform.schemas.responses import SuccessResponse
from ai_inference_optimization_platform.services.benchmark_service import (
    benchmark_service,
)
from ai_inference_optimization_platform.services.embedding_service import (
    EmbeddingService,
)
from ai_inference_optimization_platform.services.llm_service import LLMService
from ai_inference_optimization_platform.services.metrics_service import (
    metrics_service,
)
from ai_inference_optimization_platform.utils.hashing import generate_prompt_hash
from ai_inference_optimization_platform.utils.prompt_builder import PromptBuilder

# Servis Başlatmaları
redis_client = RedisClient()
llm_service = LLMService()
embedding_service = EmbeddingService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Uygulama yaşam döngüsü. Sunucu başlarken ve kapanırken çalışacak işlemleri yönetir.
    """
    logger.info("Application starting...")

    # Redis Bağlantı Kontrolü
    pong = await redis_client.ping()
    logger.info(f"Redis connected: {pong}")

    # Veritabanı (SQLite) Kurulumu
    await MetricsDatabase.initialize()
    logger.info("Application telemetry database ready.")

    yield  # Uygulama bu noktada çalışmaya başlar

    # Sunucu kapanırken çalışacak kodlar buraya eklenebilir
    logger.info("Application shutting down.")


# FastAPI Uygulama Tanımı
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
)

# Middleware ve Hata Yakalayıcılar
register_exception_handlers(app)
app.add_middleware(RequestTimerMiddleware)


# ==========================================
# ENDPOINT'LER
# ==========================================

@app.get("/", response_model=SuccessResponse)
async def root():
    logger.info("Root endpoint called.")
    return SuccessResponse(data={"message": settings.app_name})


@app.get("/health", response_model=SuccessResponse)
async def health():
    logger.info("Health endpoint called.")
    return SuccessResponse(data={"status": "healthy"})


@app.post("/generate")
async def generate_stream_endpoint(request: GenerateRequest):
    """
    Endpoint that handles the prompt and streams back the LLM or Cache response.
    """
    async def event_generator():
        # ✨ GÜNCELLENDİ: Frontend'den gelen dinamik parametreleri LLMService'e iletiyoruz
        async for chunk in llm_service.generate_stream(
            prompt=request.prompt,
            provider_override=request.provider,
            model_override=request.model_name,
            api_key=request.api_key
        ):
            yield chunk

    return StreamingResponse(event_generator(), media_type="text/plain")

@app.get("/metrics", response_model=SuccessResponse)
async def metrics():
    """
    Anlık bellek içi (in-memory) metrikleri döner.
    """
    return SuccessResponse(
        data={
            **metrics_service.get_metrics(),
            **benchmark_service.get_metrics(),
        }
    )


@app.get("/analytics/history")
async def get_inference_history(limit: int = 50):
    """
    Returns historical telemetry and performance metrics of AI inferences from SQLite.
    """
    history = await MetricsDatabase.get_history(limit=limit)
    return {
        "status": "success",
        "count": len(history),
        "data": history
    }


@app.get("/test")
async def test():
    """
    Hata yakalama mekanizmasını (Exception Handler) test etmek içindir.
    """
    raise AIInferenceException("This is a test exception.")