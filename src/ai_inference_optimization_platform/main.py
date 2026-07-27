from contextlib import asynccontextmanager
from fastapi.responses import StreamingResponse

from fastapi import FastAPI

from ai_inference_optimization_platform.cache.redis_client import RedisClient
from ai_inference_optimization_platform.config.settings import settings
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
from ai_inference_optimization_platform.schemas.requests import (
    GenerateRequest,
)
from ai_inference_optimization_platform.schemas.responses import (
    SuccessResponse,
)
from ai_inference_optimization_platform.services.llm_service import LLMService
from ai_inference_optimization_platform.utils.hashing import (
    generate_prompt_hash,
)
from ai_inference_optimization_platform.services.metrics_service import metrics_service
from ai_inference_optimization_platform.services.embedding_service import (
    EmbeddingService,
)
from ai_inference_optimization_platform.services.benchmark_service import (
    benchmark_service,
)

redis_client = RedisClient()
llm_service = LLMService()
embedding_service = EmbeddingService()


@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info("Application starting...")

    pong = await redis_client.ping()

    logger.info(f"Redis connected: {pong}")

    yield

    logger.info("Application shutting down.")

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    lifespan=lifespan,
)

register_exception_handlers(app)
app.add_middleware(RequestTimerMiddleware)


@app.get("/", response_model=SuccessResponse)
async def root():

    logger.info("Root endpoint called.")

    return SuccessResponse(
        data={
            "message": settings.app_name
        }
    )


@app.get("/health", response_model=SuccessResponse)
async def health():

    logger.info("Health endpoint called.")

    return SuccessResponse(
        data={
            "status": "healthy"
        }
    )


@app.get("/test")
async def test():

    raise AIInferenceException("This is a test exception.")


@app.post("/generate")
async def generate_stream_endpoint(request: GenerateRequest): # Kendi Pydantic request modelinin ismini kullan
    """
    Endpoint that handles the prompt and streams back the LLM or Cache response.
    """
    async def event_generator():
        async for chunk in llm_service.generate_stream(prompt=request.prompt):
            yield chunk
            
    # media_type="text/plain" veya akış formatına göre "text/event-stream" kullanabilirsin.
    return StreamingResponse(event_generator(), media_type="text/plain")


@app.get("/metrics", response_model=SuccessResponse)
async def metrics():

    return SuccessResponse(
        data={
            **metrics_service.get_metrics(),
            **benchmark_service.get_metrics(),
        }
    )