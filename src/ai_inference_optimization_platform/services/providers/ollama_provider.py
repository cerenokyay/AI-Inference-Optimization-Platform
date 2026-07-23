import httpx

from ai_inference_optimization_platform.config.settings import settings
from ai_inference_optimization_platform.logging.logger import logger
from ai_inference_optimization_platform.services.providers.base_provider import (
    BaseProvider,
)


class OllamaProvider(BaseProvider):
    """
    Provider responsible for communicating with the local Ollama server.
    """

    def __init__(self) -> None:
        self.base_url = "http://localhost:11434"

    async def generate(self, prompt: str) -> str:
        logger.info("Sending request to Ollama.")

        payload = {
            "model": settings.default_model,
            "prompt": prompt,
            "stream": False,
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.base_url}/api/generate",
                json=payload,
            )

        response.raise_for_status()

        data = response.json()

        logger.info("Response received from Ollama.")

        return data["response"]