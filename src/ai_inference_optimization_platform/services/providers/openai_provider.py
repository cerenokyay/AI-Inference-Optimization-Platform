from typing import AsyncGenerator

from fastapi import HTTPException
from openai import AsyncOpenAI

from ai_inference_optimization_platform.config.settings import settings
from ai_inference_optimization_platform.logging.logger import logger
from ai_inference_optimization_platform.services.providers.base_provider import (
    BaseProvider,
)


class OpenAIProvider(BaseProvider):
    """OpenAI API provider with streaming and error handling support."""

    def __init__(self) -> None:
        if not settings.openai_api_key:
            logger.error("OpenAI API key is missing.")
            raise ValueError("OPENAI_API_KEY is not set in .env")
            
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        # Eğer ayarlarda gpt geçmiyorsa varsayılanı kullan
        self.model = settings.default_model if "gpt" in settings.default_model else "gpt-3.5-turbo"

    async def generate(self, prompt: str) -> str:
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"OpenAI connection error: {e}")
            raise HTTPException(
                status_code=503,
                detail="OpenAI Provider is currently unreachable.",
            )

    async def generate_stream(self, prompt: str) -> AsyncGenerator[str, None]:
        try:
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
            )
            async for chunk in stream:
                content = chunk.choices[0].delta.content
                if content is not None:
                    yield content
        except Exception as e:
            logger.error(f"OpenAI streaming error: {e}")
            raise HTTPException(
                status_code=503,
                detail="OpenAI Provider streaming is currently unreachable.",
            )