import json
import os

import faiss
import numpy as np

from ai_inference_optimization_platform.config.settings import settings
from ai_inference_optimization_platform.logging.logger import logger


class FAISSStore:
    """
    Vector store backed by FAISS.

    Responsibilities:
    - Store embeddings
    - Search similar embeddings
    - Prevent duplicate vectors
    - Persist vectors and metadata
    """

    def __init__(self, dimension: int = 768) -> None:
        self.dimension = dimension

        # Inner Product + normalized vectors = Cosine Similarity
        self.index = faiss.IndexFlatIP(dimension)

        self.metadata: list[dict] = []

        logger.info(
            f"FAISSStore initialized. Dimension={dimension}"
        )

    def add(
        self,
        embedding: list[float],
        prompt: str,
        response: str,
    ) -> bool:
        """
        Add a new vector if it is not already present.

        Returns:
            True  -> inserted
            False -> duplicate skipped
        """

        if self.exists(
            embedding,
            settings.duplicate_threshold,
        ):
            logger.info(
                "Duplicate embedding detected. Skipping insert."
            )
            return False

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

        logger.info(
            f"FAISS entries: {self.index.ntotal}"
        )

        return True

    def exists(
        self,
        embedding: list[float],
        threshold: float,
    ) -> bool:
        """
        Returns True if a very similar embedding
        already exists inside the vector store.
        """

        if self.index.ntotal == 0:
            return False

        vector = np.array(
            [embedding],
            dtype=np.float32,
        )

        faiss.normalize_L2(vector)

        scores, indices = self.index.search(
            vector,
            1,
        )

        score = float(scores[0][0])
        index = int(indices[0][0])

        logger.info(
            f"Duplicate check score: {score:.4f}"
        )

        return (
            index != -1
            and score >= threshold
        )

    def search(
        self,
        embedding: list[float],
        top_k: int = 1,
    ):
        """
        Search for the nearest vectors.

        Returns:
            (scores, indices) or None
        """

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

    def save(
        self,
        path: str,
    ) -> None:
        """
        Persist index and metadata.
        """

        os.makedirs(
            os.path.dirname(path),
            exist_ok=True,
        )

        faiss.write_index(
            self.index,
            path,
        )

        metadata_path = path.replace(
            ".faiss",
            ".json",
        )

        with open(
            metadata_path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                self.metadata,
                file,
                ensure_ascii=False,
                indent=4,
            )

        logger.info(
            f"Saved {self.index.ntotal} vectors and metadata to disk -> {path}"
        )

    def load(
        self,
        path: str,
    ) -> None:
        """
        Restore index and metadata from disk.
        """

        if not os.path.exists(path):
            logger.info(
                "No existing FAISS index found."
            )
            return

        self.index = faiss.read_index(path)

        self.dimension = self.index.d

        metadata_path = path.replace(
            ".faiss",
            ".json",
        )

        if os.path.exists(metadata_path):

            with open(
                metadata_path,
                "r",
                encoding="utf-8",
            ) as file:
                self.metadata = json.load(file)

        logger.info(
            "FAISS index loaded. "
            f"Total vectors: {self.index.ntotal}, "
            f"Total metadata entries: {len(self.metadata)}"
        )