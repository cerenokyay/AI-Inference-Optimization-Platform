from ai_inference_optimization_platform.services.embeddings.mock_embedding_provider import (
    MockEmbeddingProvider,
)


class EmbeddingFactory:

    @staticmethod
    def create():

        return MockEmbeddingProvider()