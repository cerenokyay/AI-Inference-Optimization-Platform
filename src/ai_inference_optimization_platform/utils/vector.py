from math import sqrt


def cosine_similarity(
    vector1: list[float],
    vector2: list[float],
) -> float:
    """
    Calculate cosine similarity between two vectors.
    """

    if len(vector1) != len(vector2):
        raise ValueError("Vectors must have the same length.")

    dot_product = sum(
        a * b
        for a, b in zip(vector1, vector2)
    )

    magnitude1 = sqrt(
        sum(a * a for a in vector1)
    )

    magnitude2 = sqrt(
        sum(b * b for b in vector2)
    )

    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    return dot_product / (magnitude1 * magnitude2)