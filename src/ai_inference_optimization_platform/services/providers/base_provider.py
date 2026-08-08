from abc import ABC, abstractmethod
from typing import AsyncGenerator


class BaseProvider(ABC):
    """Abstract base class for all LLM providers."""

    @abstractmethod
    async def generate(self, prompt: str) -> str:
        """Generates a complete response string."""
        pass

    @abstractmethod
    async def generate_stream(self, prompt: str) -> AsyncGenerator[str, None]:
        """Streams the response chunk by chunk."""
        pass