from typing import AsyncGenerator

from fastapi import HTTPException

from ai_inference_optimization_platform.logging.logger import logger
from ai_inference_optimization_platform.services.providers.base_provider import (
    BaseProvider,
)


class FallbackProvider(BaseProvider):
    """A composite provider that tries multiple LLM providers in sequence."""

    def __init__(self, providers: list[BaseProvider]) -> None:
        self.providers = providers

    async def generate(self, prompt: str) -> str:
        errors = []
        for provider in self.providers:
            provider_name = provider.__class__.__name__
            try:
                logger.info(f"Attempting generation with {provider_name}...")
                return await provider.generate(prompt)
            except Exception as e:
                logger.warning(
                    f"{provider_name} failed: {e}. Falling back to next provider..."
                )
                errors.append(f"{provider_name}: {str(e)}")

        logger.error("All providers in the fallback chain failed.")
        raise HTTPException(
            status_code=503,
            detail=f"All LLM providers failed. Errors: {errors}",
        )

    async def generate_stream(self, prompt: str) -> AsyncGenerator[str, None]:
        errors = []
        for provider in self.providers:
            provider_name = provider.__class__.__name__
            try:
                logger.info(f"Attempting streaming generation with {provider_name}...")

                # Sağlayıcının stream fonksiyonunu başlat
                stream_generator = provider.generate_stream(prompt)

                async for chunk in stream_generator:
                    yield chunk

                # Eğer ilk chunk başarıyla geldiyse ve döngü bittiyse işlemi sonlandır (diğerine geçme)
                return

            except Exception as e:
                logger.warning(
                    f"{provider_name} streaming failed: {e}. Falling back..."
                )
                errors.append(f"{provider_name}: {str(e)}")

        logger.error("All providers in the fallback chain failed during streaming.")
        raise HTTPException(
            status_code=503, 
            detail=f"All LLM providers failed. Errors: {errors}"
        )