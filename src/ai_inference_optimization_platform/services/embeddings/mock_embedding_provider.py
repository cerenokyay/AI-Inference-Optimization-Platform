from ai_inference_optimization_platform.services.embeddings.embedding_provider import (
    EmbeddingProvider,
)


class MockEmbeddingProvider(EmbeddingProvider):
    """
    Fake embedding provider used for testing.
    """

    async def generate_embedding(
        self,
        text: str,
    ) -> list[float]:

        return [
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
        ]