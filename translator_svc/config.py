"""Runtime settings, read from environment variables."""

from typing import Annotated, Literal

from fastapi import Request
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    # Fields can be set by name in code, or by their env-var alias.
    model_config = SettingsConfigDict(populate_by_name=True)

    # Config is logged -- do not store secrets here!
    log_level: str = "INFO"

    # ---- Translator-specific settings ----
    # LLM backend
    llm_backend: Literal["llama_cpp", "fake"] = Field(
        "llama_cpp", validation_alias="LLM_BACKEND"
    )
    llm_server_url: str = Field(
        "http://localhost:8080", validation_alias="LLM_SERVER_URL"
    )
    llm_model_variant: Literal["e4b", "e2b"] = Field(
        "e4b",
        validation_alias="LLM_MODEL_VARIANT",
        description="Gemma 4 variant: e4b (larger, ~2.5GB) or e2b (smaller, ~1.3GB).",
    )
    llm_timeout_seconds: float = Field(30.0, validation_alias="LLM_TIMEOUT_SECONDS")
    llm_health_check_interval_seconds: float = Field(
        10.0, validation_alias="LLM_HEALTH_CHECK_INTERVAL_SECONDS"
    )
    # Thinking Mode is OFF (the system prompt no longer emits <|think|>), so the
    # budget only has to cover the two-line answer. 256 is generous for short
    # conversational phrases; raise via LLM_MAX_TOKENS only if long inputs get
    # truncated (empty content is still handled as a no-speech result).
    llm_max_tokens: int = Field(256, validation_alias="LLM_MAX_TOKENS")
    # Sampling for Gemma 4 with reasoning off. Temperature 1.0 (matching the
    # llama-server --temp default). Verified against ground-truth audio: on a
    # hard/ambiguous clip, LOWER temps converged confidently on the WRONG
    # transcription (temp 0.0 repeated one wrong reading 7/7), while temp 1.0's
    # exploration was the only setting that recovered the true phrase. For audio
    # transcription, a confident-but-wrong result is the worst outcome, so we
    # favor the higher temp. min_p still keeps the nucleus narrow enough to
    # exclude low-probability phonetic siblings (keeps tokens with prob >=
    # min_p * max_prob, adapting its cutoff to the model's confidence).
    llm_temperature: float = Field(1.0, validation_alias="LLM_TEMPERATURE")
    llm_min_p: float = Field(0.05, validation_alias="LLM_MIN_P")
    # Repeat penalty discourages the repetition loops Gemma can fall into without
    # distorting natural word reuse.
    llm_repeat_penalty: float = Field(1.10, validation_alias="LLM_REPEAT_PENALTY")

    # TTS backend
    tts_backend: Literal["kokoro_local", "fake"] = Field(
        "kokoro_local", validation_alias="TTS_BACKEND"
    )
    tts_model_path: str = Field("./models/kokoro", validation_alias="TTS_MODEL_PATH")

    # Session store backend
    session_store_backend: Literal["memory", "redis"] = Field(
        "memory", validation_alias="SESSION_STORE_BACKEND"
    )
    session_max_phrases: int = Field(10, validation_alias="SESSION_MAX_PHRASES")
    session_ttl_seconds: int = Field(1800, validation_alias="SESSION_TTL_SECONDS")
    session_prompt_truncate_chars: int = Field(
        200, validation_alias="SESSION_PROMPT_TRUNCATE_CHARS"
    )
    redis_url: str = Field("", validation_alias="REDIS_URL")

    # Languages
    # Languages are identified by their plain English name (e.g. "English",
    # "Spanish"). The name is the single identifier used end-to-end: request
    # body, validation, TTS voice lookup, and session storage. Regional/dialect
    # nuance (e.g. Latin American Spanish) is applied in the LLM prompt via the
    # PromptBuilder's dialect-hint map, not encoded into the identifier.
    supported_languages: set[str] = Field(
        default_factory=lambda: {
            "English",
            "Spanish",
            "Hindi",
            "French",
            "Italian",
        },
        validation_alias="SUPPORTED_LANGUAGES",
    )

    # Audio limits
    max_audio_bytes: int = Field(5 * 1024 * 1024, validation_alias="MAX_AUDIO_BYTES")

    # Validation thresholds
    validation_min_ratio: float = Field(0.3, validation_alias="VALIDATION_MIN_RATIO")
    validation_max_ratio: float = Field(4.0, validation_alias="VALIDATION_MAX_RATIO")
    validation_refusal_patterns: list[str] = Field(
        default_factory=lambda: [
            "i cannot",
            "i can't translate",
            "as an ai",
            "no puedo",
        ],
    )

    # CORS — origins allowed to call the API from a browser. Comma-separated
    # when set via the CORS_ALLOW_ORIGINS env var. NoDecode skips JSON parsing
    # so the comma-splitting validator below receives the raw string.
    cors_allow_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=list, validation_alias="CORS_ALLOW_ORIGINS"
    )
    # Regex alternative to cors_allow_origins. Defaults to localhost on any port
    # so a locally served UI can call the API; set it to "" to disable.
    cors_allow_origin_regex: str = Field(
        r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
        validation_alias="CORS_ALLOW_ORIGIN_REGEX",
    )

    @field_validator("cors_allow_origins", mode="before")
    @classmethod
    def _split_cors_origins(cls, value: object) -> object:
        """Allow CORS_ALLOW_ORIGINS to be a comma-separated string."""
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value


def get_settings(request: Request) -> Settings:
    return request.app.state.config
