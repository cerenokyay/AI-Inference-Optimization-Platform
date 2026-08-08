from pydantic import BaseModel, Field
from typing import Optional


class GenerateRequest(BaseModel):
    """
    Request model for text generation.
    """

    prompt: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="Prompt sent to the language model.",
    )

    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature.",
    )

    max_tokens: int = Field(
        default=256,
        ge=1,
        le=4096,
        description="Maximum number of generated tokens.",
    )

    # ==========================================
    # ✨ YENİ: Dinamik Model Yönlendirme Alanları
    # ==========================================
    provider: Optional[str] = Field(
        default=None, 
        description="Örn: openai, ollama, anthropic"
    )
    
    model_name: Optional[str] = Field(
        default=None, 
        description="Örn: gpt-4o, qwen2.5:3b"
    )
    
    api_key: Optional[str] = Field(
        default=None, 
        description="Kullanıcıya özel API anahtarı"
    )