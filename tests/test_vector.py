from ai_inference_optimization_platform.utils.vector import (
    cosine_similarity,
)


def test_cosine_similarity():

    vector1 = [1, 2, 3]
    vector2 = [1, 2, 3]

    similarity = cosine_similarity(
        vector1,
        vector2,
    )

    assert similarity == 1.0