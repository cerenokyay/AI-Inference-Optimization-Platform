import json
from typing import AsyncGenerator

import httpx
from fastapi import HTTPException

from ai_inference_optimization_platform.config.settings import settings
from ai_inference_optimization_platform.logging.logger import logger
from ai_inference_optimization_platform.services.providers.base_provider import (
    BaseProvider,
)


class OllamaProvider(BaseProvider):
    """Ollama API provider with streaming and error handling support."""

    def __init__(self) -> None:
        self.base_url = "http://localhost:11434/api/generate"
        self.model = settings.default_model

    async def generate(self, prompt: str) -> str:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.base_url,
                    json={"model": self.model, "prompt": prompt, "stream": False},
                    timeout=60.0,
                )
                response.raise_for_status()
                return response.json().get("response", "")
        except httpx.ConnectError as e:
            logger.error(f"Ollama connection error: {e}")
            raise HTTPException(
                status_code=503,
                detail="LLM Provider is currently unreachable. Please ensure Ollama is running.",
            )

    async def generate_stream(self, prompt: str) -> AsyncGenerator[str, None]:
        """Streams the response from Ollama token by token."""
        try:
            async with httpx.AsyncClient() as client:
                async with client.stream(
                    "POST",
                    self.base_url,
                    json={"model": self.model, "prompt": prompt, "stream": True},
                    timeout=60.0,
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line:
                            data = json.loads(line)
                            if "response" in data:
                                yield data["response"]
        except httpx.ConnectError as e:
            logger.error(f"Ollama connection error: {e}")
            raise HTTPException(
                status_code=503,
                detail="LLM Provider is currently unreachable.",
            )