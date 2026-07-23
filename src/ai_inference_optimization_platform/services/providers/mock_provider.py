from ai_inference_optimization_platform.logging.logger import logger
from ai_inference_optimization_platform.services.providers.base_provider import (
    BaseProvider,
)


class MockProvider(BaseProvider):
    """
    Mock provider used during development.
    """

    async def generate(self, prompt: str) -> str:
        logger.info("Mock provider generating response.")

        return f"Mock response for: {prompt}"