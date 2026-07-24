import hashlib


def generate_prompt_hash(prompt: str) -> str:
    """
    Generate a deterministic SHA256 hash for a prompt.
    """

    return hashlib.sha256(
        prompt.encode("utf-8")
    ).hexdigest()