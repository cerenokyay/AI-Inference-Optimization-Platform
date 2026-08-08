from typing import AsyncGenerator

from anthropic import AsyncAnthropic
from fastapi import HTTPException

from ai_inference_optimization_platform.config.settings import settings
from ai_inference_optimization_platform.logging.logger import logger
from ai_inference_optimization_platform.services.providers.base_provider import (
    BaseProvider,
)


class AnthropicProvider(BaseProvider):
    """Anthropic API provider with streaming and error handling support."""

    def __init__(self) -> None:
        if not settings.anthropic_api_key:
            logger.error("Anthropic API key is missing.")
            raise ValueError("ANTHROPIC_API_KEY is not set in .env")
            
        self.client = AsyncAnthropic(api_key=settings.anthropic_api_key)
        self.model = settings.default_model if "claude" in settings.default_model else "claude-3-haiku-20240307"

    async def generate(self, prompt: str) -> str:
        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Anthropic connection error: {e}")
            raise HTTPException(
                status_code=503,
                detail="Anthropic Provider is currently unreachable.",
            )

    async def generate_stream(self, prompt: str) -> AsyncGenerator[str, None]:
        try:
            async with self.client.messages.stream(
                model=self.model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                async for text in stream.text_stream:
                    yield text
        except Exception as e:
            logger.error(f"Anthropic streaming error: {e}")
            raise HTTPException(
                status_code=503,
                detail="Anthropic Provider streaming is currently unreachable.",
            )