from fastapi import FastAPI
from ai_inference_optimization_platform.exceptions.handlers import (
    register_exception_handlers,
)
from ai_inference_optimization_platform.schemas.responses import (
    SuccessResponse,
)
from ai_inference_optimization_platform.config.settings import settings
from ai_inference_optimization_platform.logging.logger import logger
from ai_inference_optimization_platform.exceptions.custom_exceptions import (
    AIInferenceException,
)
from ai_inference_optimization_platform.services.llm_service import LLMService


logger.info("Application started successfully.")
llm_service = LLMService()

app = FastAPI(

    title="AI Inference Optimization Platform",
    version="0.1.0",
)

register_exception_handlers(app)

@app.get("/", response_model=SuccessResponse)
async def root():

    logger.info("Root endpoint called.")

    return SuccessResponse(
        data={
            "message": "AI Inference Optimization Platform"
        }
    )


@app.get("/health", response_model=SuccessResponse)
async def health():

    logger.info("Health check requested.")

    return SuccessResponse(
        data={
            "status": "healthy"
        }
    )

@app.get("/test")
async def test():
    raise AIInferenceException("This is a test exception.")


@app.get("/generate")
async def generate():

    response = await llm_service.generate(
        prompt="Hello AI"
    )

    return SuccessResponse(
        data={
            "response": response
        }
    )