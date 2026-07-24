import re


def normalize_prompt(prompt: str) -> str:
    """
    Normalize user prompt before hashing.
    """

    prompt = prompt.lower()

    prompt = prompt.strip()

    prompt = re.sub(r"\s+", " ", prompt)

    return prompt