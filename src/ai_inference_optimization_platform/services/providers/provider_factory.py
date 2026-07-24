from ai_inference_optimization_platform.config.settings import settings

from ai_inference_optimization_platform.services.providers.mock_provider import (
    MockProvider,
)

from ai_inference_optimization_platform.services.providers.ollama_provider import (
    OllamaProvider,
)


class ProviderFactory:

    @staticmethod
    def create():

        provider = settings.default_provider.lower()

        if provider == "ollama":
            return OllamaProvider()

        if provider == "mock":
            return MockProvider()

        raise ValueError(
            f"Unknown provider: {provider}"
        )