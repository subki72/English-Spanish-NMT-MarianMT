"""Pydantic v2 schemas for API requests, responses, and health status.
Implements BLK-04 input sanitization (length caps, whitespace check, control char filtering).
"""

import unicodedata

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.config import settings


def sanitize_and_validate_text(text: str, max_chars: int | None = None) -> str:
    """Validate and sanitize a single input string.

    Guards against:
    - Empty or whitespace-only strings
    - Unbounded payload DoS (> max_chars)
    - Null bytes and unprintable control characters (except \\t, \\n, \\r)
    - Leading UTF-8 BOM artifacts
    - Retains valid compound emojis with Zero-Width Joiners (ZWJ)
    """
    if max_chars is None:
        max_chars = settings.max_input_length

    if not isinstance(text, str):
        raise ValueError("Input text must be a string.")

    # Strip optional UTF-8 Byte Order Mark
    clean_text = text.lstrip("\ufeff")

    stripped = clean_text.strip()
    if not stripped:
        raise ValueError("Input text cannot be empty or solely whitespace.")

    if len(stripped) > max_chars:
        raise ValueError(
            f"Input text exceeds maximum allowed length of {max_chars} characters (received {len(stripped)})."
        )

    # Filter/reject null bytes and illicit ASCII/Unicode control characters (Cc / Cs)
    for idx, ch in enumerate(clean_text):
        if ch in ("\n", "\r", "\t"):
            continue
        category = unicodedata.category(ch)
        if category in ("Cc", "Cs"):  # Control or Surrogate characters
            raise ValueError(
                f"Input text contains invalid control or unprintable character at position {idx} (U+{ord(ch):04X})."
            )

    return stripped


class TranslationRequest(BaseModel):
    """Translation request payload accepting either a single string or a list of strings."""

    text: str | list[str] = Field(
        ...,
        description="Single English sentence or list of sentences to translate to Spanish (max characters per sentence configured by max_input_length).",
    )
    beam_size: int | None = Field(
        default=None,
        ge=1,
        le=10,
        description="Optional beam size override (default: configured server beam_size).",
    )
    max_decoding_length: int | None = Field(
        default=None,
        ge=16,
        le=512,
        description="Optional maximum token decoding length override (default: configured server max_decoding_length).",
    )

    @field_validator("text")
    @classmethod
    def validate_text_field(cls, v: str | list[str]) -> list[str]:
        if isinstance(v, str):
            validated = [sanitize_and_validate_text(v, max_chars=settings.max_input_length)]
        elif isinstance(v, list):
            if len(v) == 0:
                raise ValueError("Text list must contain at least one sentence.")
            if len(v) > 64:
                raise ValueError(f"Batch request exceeds maximum allowed items (max 64, received {len(v)}).")
            validated = [sanitize_and_validate_text(item, max_chars=settings.max_input_length) for item in v]
        else:
            raise ValueError("Field 'text' must be either a string or a list of strings.")
        return validated

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "text": "Artificial intelligence is changing the world.",
                    "beam_size": 4,
                },
                {
                    "text": [
                        "Law is mostly about logic.",
                        "I want to share an idea with you today.",
                    ],
                },
            ]
        }
    )


class TranslationItem(BaseModel):
    """Individual translation item pair."""

    source: str = Field(..., description="Original English input sentence.")
    target: str = Field(..., description="Translated Spanish output sentence.")


class TranslationResponse(BaseModel):
    """Translation response structure with metadata and latency timing."""

    translations: list[TranslationItem] = Field(..., description="List of source and translated text pairs.")
    source_language: str = Field(default="en", description="Source ISO language code.")
    target_language: str = Field(default="es", description="Target ISO language code.")
    model: str = Field(..., description="Model identifier and quantization format.")
    latency_ms: float = Field(..., description="End-to-end translation latency in milliseconds.")
    tokens_generated: int = Field(default=0, description="Total tokens generated across all translated sentences.")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "translations": [
                    {
                        "source": "Artificial intelligence is changing the world.",
                        "target": "La inteligencia artificial está cambiando el mundo.",
                    }
                ],
                "source_language": "en",
                "target_language": "es",
                "model": "Helsinki-NLP/opus-mt-en-es-ct2-int8",
                "latency_ms": 18.42,
                "tokens_generated": 10,
            }
        }
    )


class HealthResponse(BaseModel):
    """Liveness probe response."""

    status: str = Field(default="ok", description="Liveness status ('ok').")
    version: str = Field(..., description="Service version.")
    uptime_seconds: float = Field(..., description="Service uptime in seconds.")


class ReadyResponse(BaseModel):
    """Readiness probe response indicating whether model and batcher are initialized."""

    ready: bool = Field(..., description="True if service is ready to accept traffic.")
    model_loaded: bool = Field(..., description="True if CTranslate2 model is loaded.")
    tokenizer_loaded: bool = Field(..., description="True if SentencePiece processor is loaded.")
    batcher_running: bool = Field(default=True, description="True if dynamic batcher worker is active.")
    details: str | None = Field(default=None, description="Diagnostic details if not ready.")
