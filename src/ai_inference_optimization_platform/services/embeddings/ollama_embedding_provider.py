import httpx

from ai_inference_optimization_platform.services.embeddings.embedding_provider import (
    EmbeddingProvider,
)


class OllamaEmbeddingProvider(EmbeddingProvider):

    async def generate_embedding(
        self,
        text: str,
    ) -> list[float]:

        async with httpx.AsyncClient() as client:

            response = await client.post(
                "http://localhost:11434/api/embeddings",
                json={
                    "model": "nomic-embed-text",
                    "prompt": text,
                },
            )

        return response.json()["embedding"]