from abc import ABC, abstractmethod


class BaseProvider(ABC):
    """
    Base interface for all LLM providers.
    """

    @abstractmethod
    async def generate(self, prompt: str) -> str:
        """
        Generate a response from a language model.
        """
        pass