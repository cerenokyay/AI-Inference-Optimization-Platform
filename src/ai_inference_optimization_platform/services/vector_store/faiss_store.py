import json
import os

import faiss
import numpy as np

from ai_inference_optimization_platform.logging.logger import logger


class FAISSStore:
    """Stores embeddings and metadata inside a FAISS vector index with JSON persistence."""

    def __init__(self, dimension: int = 768) -> None:
        self.dimension = dimension

        # Cosine similarity için normalize edilmiş Inner Product
        self.index = faiss.IndexFlatIP(dimension)

        # Response metadata listesi
        self.metadata = []

        logger.info(f"FAISSStore initialized. Dimension={dimension}")

    def add(
        self,
        embedding: list[float],
        prompt: str,
        response: str,
    ) -> None:
        # Duplicate (Mükerrer Kayıt) Kontrolü
        for item in self.metadata:
            if item["prompt"] == prompt:
                logger.info("Prompt already exists in vector store. Skipping add.")
                return

        vector = np.array(
            [embedding],
            dtype=np.float32,
        )

        faiss.normalize_L2(vector)

        self.index.add(vector)

        self.metadata.append(
            {
                "prompt": prompt,
                "response": response,
            }
        )

        logger.info(f"FAISS entries: {self.index.ntotal}")

    def search(
        self,
        embedding: list[float],
        top_k: int = 5,
    ):
        if self.index.ntotal == 0:
            return None

        vector = np.array(
            [embedding],
            dtype=np.float32,
        )

        faiss.normalize_L2(vector)

        scores, indices = self.index.search(
            vector,
            top_k,
        )

        return scores, indices

    def save(self, path: str) -> None:
        """Save FAISS index and metadata JSON to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # 1. FAISS index'i diske yaz
        faiss.write_index(self.index, path)

        # 2. Metadata'yı JSON olarak yaz
        metadata_path = path.replace(".faiss", ".json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(
                self.metadata,
                f,
                ensure_ascii=False,
                indent=2,
            )

        logger.info(f"Saved {self.index.ntotal} vectors and metadata to disk -> {path}")

    def load(self, path: str) -> None:
        """Load FAISS index and metadata JSON from disk."""
        if not os.path.exists(path):
            logger.info("No existing FAISS index found.")
            return

        # 1. Index'i yükle
        self.index = faiss.read_index(path)
        self.dimension = self.index.d

        # 2. Metadata'yı yükle
        metadata_path = path.replace(".faiss", ".json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = []

        logger.info(
            f"FAISS index loaded. Total vectors: {self.index.ntotal}, Total metadata entries: {len(self.metadata)}"
        )