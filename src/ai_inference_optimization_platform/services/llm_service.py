from ai_inference_optimization_platform.logging.logger import logger
from ai_inference_optimization_platform.services.providers.ollama_provider import OllamaProvider


class LLMService:
    """
    Service responsible for interacting with language model providers.
    """

    def __init__(self) -> None:
        self.provider = OllamaProvider()
        logger.info("LLMService initialized.")

    async def generate(self, prompt: str) -> str:
        logger.info("Generating response through provider.")

        return await self.provider.generate(prompt)