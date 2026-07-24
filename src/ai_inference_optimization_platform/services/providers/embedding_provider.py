from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """
    Base embedding provider.
    """

    @abstractmethod
    async def embed(
        self,
        text: str,
    ) -> list[float]:
        pass