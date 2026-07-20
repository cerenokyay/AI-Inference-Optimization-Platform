from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from ai_inference_optimization_platform.exceptions.custom_exceptions import (
    AIInferenceException,
)
from ai_inference_optimization_platform.logging.logger import logger


def register_exception_handlers(app: FastAPI) -> None:
    """
    Register all custom exception handlers.
    """

    @app.exception_handler(AIInferenceException)
    async def ai_inference_exception_handler(
        request: Request,
        exc: AIInferenceException,
    ):
        logger.error(exc.message)

        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": {
                    "code": "APPLICATION_ERROR",
                    "message": exc.message,
                },
            },
        )