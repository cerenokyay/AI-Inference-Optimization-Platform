from ai_inference_optimization_platform.config.settings import settings
from ai_inference_optimization_platform.logging.logger import logger
from ai_inference_optimization_platform.services.providers.anthropic_provider import (
    AnthropicProvider,
)
from ai_inference_optimization_platform.services.providers.base_provider import (
    BaseProvider,
)
from ai_inference_optimization_platform.services.providers.fallback_provider import (
    FallbackProvider,
)
from ai_inference_optimization_platform.services.providers.ollama_provider import (
    OllamaProvider,
)
from ai_inference_optimization_platform.services.providers.openai_provider import (
    OpenAIProvider,
)


class ProviderFactory:
    """Factory class to create and return the configured LLM provider or a fallback chain."""

    @staticmethod
    def create() -> BaseProvider:
        chain_str = settings.provider_fallback_chain
        provider_names = [p.strip().lower() for p in chain_str.split(",")]

        logger.info(f"Initializing Provider Chain: {provider_names}")

        provider_map = {
            "openai": OpenAIProvider,
            "anthropic": AnthropicProvider,
            "ollama": OllamaProvider,
        }

        active_providers = []
        for name in provider_names:
            if name in provider_map:
                try:
                    # API Key eksikse ValueError fırlatacak, try-except ile atlıyoruz.
                    provider_instance = provider_map[name]()
                    active_providers.append(provider_instance)
                except ValueError as e:
                    logger.warning(f"Skipping {name.upper()} in fallback chain: {e}")
            else:
                logger.warning(f"Unknown provider '{name}' in fallback chain.")

        if not active_providers:
            logger.error("No valid providers initialized! Defaulting to Ollama.")
            return OllamaProvider()

        # Eğer zincirde tek bir geçerli sağlayıcı varsa doğrudan onu dön
        if len(active_providers) == 1:
            return active_providers[0]

        # Birden fazla sağlayıcı varsa Fallback zincirini dön
        return FallbackProvider(active_providers)