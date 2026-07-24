from contextlib import asynccontextmanager

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

redis_client = RedisClient()
llm_service = LLMService()


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


@app.post("/generate", response_model=SuccessResponse)
async def generate(request: GenerateRequest):

    logger.info("Generate endpoint called.")

    prompt_hash = generate_prompt_hash(request.prompt)

    logger.info(f"Prompt hash: {prompt_hash}")
    

    response = await llm_service.generate(
        prompt=request.prompt
    )

    return SuccessResponse(
        data={
            "response": response
        }
    )

@app.get("/metrics", response_model=SuccessResponse)
async def metrics():

    return SuccessResponse(
        data=metrics_service.get_metrics()
    )