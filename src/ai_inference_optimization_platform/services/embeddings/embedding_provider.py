from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """
    Base class for embedding providers.
    """

    @abstractmethod
    async def generate_embedding(
        self,
        text: str,
    ) -> list[float]:
        pass