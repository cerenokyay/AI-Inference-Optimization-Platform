from pathlib import Path
import pickle

import faiss

from ai_inference_optimization_platform.logging.logger import logger

DATA_DIR = Path("data/faiss")

INDEX_FILE = DATA_DIR / "index.faiss"
METADATA_FILE = DATA_DIR / "metadata.pkl"


def ensure_directory() -> None:
    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def save_index(index) -> None:

    ensure_directory()

    faiss.write_index(
        index,
        str(INDEX_FILE),
    )

    logger.info("FAISS index saved.")


def load_index():

    if not INDEX_FILE.exists():
        return None

    logger.info("Loading FAISS index.")

    return faiss.read_index(
        str(INDEX_FILE),
    )


def save_metadata(metadata) -> None:

    ensure_directory()

    with open(
        METADATA_FILE,
        "wb",
    ) as f:

        pickle.dump(
            metadata,
            f,
        )

    logger.info("Metadata saved.")


def load_metadata():

    if not METADATA_FILE.exists():
        return []

    logger.info("Loading metadata.")

    with open(
        METADATA_FILE,
        "rb",
    ) as f:

        return pickle.load(f)