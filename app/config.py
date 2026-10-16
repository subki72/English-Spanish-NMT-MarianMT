"""Application configuration using Pydantic Settings v2.
Supports 12-factor configuration via environment variables.
"""

import json
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime service configuration settings."""

    # Model & Engine
    model_dir: Path = Field(
        default=Path("models/opus-mt-en-es-ct2-int8"),
        description="Directory containing CTranslate2 INT8 model files and vocabulary",
    )
    sp_source_model: Path | None = Field(
        default=None,
        description="Path to SentencePiece source model (.spm). If None, auto-detected from model_dir.",
    )
    sp_target_model: Path | None = Field(
        default=None,
        description="Path to SentencePiece target model (.spm). If None, auto-detected from model_dir.",
    )
    device: str = Field(
        default="cpu",
        description="Compute device: 'cpu' or 'cuda'",
    )
    device_index: int = Field(
        default=0,
        description="Device index when running on GPU",
    )
    compute_type: str = Field(
        default="int8",
        description="Quantization compute type: 'int8', 'int8_float16', 'float16', or 'auto'",
    )
    intra_threads: int = Field(
        default=4,
        description="Number of OpenMP threads for CTranslate2 compute execution",
    )
    inter_threads: int = Field(
        default=1,
        description="Number of parallel translation streams within CTranslate2",
    )

    # Inference & Dynamic Batcher
    batch_size: int = Field(
        default=32,
        ge=1,
        le=256,
        description="Maximum batch size dynamically formed before inference execution",
    )
    max_wait_ms: float = Field(
        default=10.0,
        ge=0.0,
        le=500.0,
        description="Maximum wait time in milliseconds for collecting requests into a batch",
    )
    beam_size: int = Field(
        default=4,
        ge=1,
        le=10,
        description="Beam search size for decoding",
    )
    max_input_length: int = Field(
        default=1000,
        ge=1,
        le=10000,
        description="Maximum allowed input character length per sentence",
    )
    max_decoding_length: int = Field(
        default=256,
        ge=16,
        le=1024,
        description="Maximum generated tokens to prevent truncation while capping compute",
    )
    max_queue_depth: int = Field(
        default=1000,
        ge=1,
        le=50000,
        description="Maximum pending requests allowed in the dynamic batching queue before 503 saturation rejection",
    )

    # Server & Operations
    host: str = Field(default="0.0.0.0", description="Bind host")
    port: int = Field(default=8000, description="Bind port")
    workers: int = Field(default=1, description="Number of worker processes")
    log_level: str = Field(default="INFO", description="Logging level")
    environment: str = Field(default="production", description="Environment: production, staging, development")
    enable_metrics: bool = Field(default=True, description="Enable Prometheus metrics endpoint")
    cors_origins: list[str] = Field(default=["*"], description="Allowed CORS origins")
    fail_fast_startup: bool = Field(
        default=True,
        description="Whether missing model files should immediately fail service startup. Set to False for container smoke testing.",
    )

    # Rate Limiting & Throttling
    enable_rate_limiting: bool = Field(
        default=True,
        description="Enable per-IP rate limiting on translation endpoint",
    )
    rate_limit_requests: int = Field(
        default=60,
        ge=1,
        description="Maximum translation requests allowed per client IP within window",
    )
    rate_limit_window_seconds: int = Field(
        default=60,
        ge=1,
        description="Time window in seconds for sliding window rate limiter",
    )
    trusted_proxies: set[str] = Field(
        default_factory=lambda: {"127.0.0.1", "::1", "localhost"},
        description="Set of trusted reverse proxy IP addresses allowed to forward client IPs via X-Forwarded-For",
    )

    @field_validator("trusted_proxies", mode="before")
    @classmethod
    def _parse_trusted_proxies(cls, v: Any) -> Any:
        if isinstance(v, str):
            v = v.strip()
            if v.startswith("[") and v.endswith("]"):
                try:
                    return json.loads(v)
                except Exception:
                    pass
            return {item.strip() for item in v.split(",") if item.strip()}
        return v

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _parse_cors_origins(cls, v: Any) -> Any:
        if isinstance(v, str):
            v = v.strip()
            if v.startswith("[") and v.endswith("]"):
                try:
                    return json.loads(v)
                except Exception:
                    pass
            return [item.strip() for item in v.split(",") if item.strip()]
        return v

    # Observability & Structured Logging
    enable_structured_logging: bool = Field(
        default=True,
        description="Enable structured JSON access and error logging",
    )

    # Metadata
    app_name: str = Field(
        default="English-Spanish NMT MarianMT Service",
        description="FastAPI Application Name",
    )
    app_version: str = Field(
        default="1.0.0",
        description="Application version",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="",
        case_sensitive=False,
        extra="ignore",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: Any,
        env_settings: Any,
        dotenv_settings: Any,
        file_secret_settings: Any,
    ) -> tuple[Any, ...]:
        for source in (env_settings, dotenv_settings):
            if hasattr(source, "decode_complex_value"):
                orig_decode = source.decode_complex_value

                def _make_decoder(orig):
                    def _custom_decoder(field_name: str, field: Any, value: Any) -> Any:
                        try:
                            return orig(field_name, field, value)
                        except Exception:
                            if field_name == "trusted_proxies" and isinstance(value, str):
                                return {x.strip() for x in value.split(",") if x.strip()}
                            if field_name == "cors_origins" and isinstance(value, str):
                                return [x.strip() for x in value.split(",") if x.strip()]
                            raise

                    return _custom_decoder

                source.decode_complex_value = _make_decoder(orig_decode)
        return (init_settings, env_settings, dotenv_settings, file_secret_settings)


# Global singleton instance
settings = Settings()
