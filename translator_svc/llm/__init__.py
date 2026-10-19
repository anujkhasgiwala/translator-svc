"""LLMClient factory."""

from translator_svc.config import Settings

from .base import LLMClient
from .fake import FakeLLMClient
from .llama_cpp import LlamaCppLLMClient

__all__ = ["LLMClient", "make_llm_client"]


def make_llm_client(settings: Settings) -> LLMClient:
    backend = settings.llm_backend
    if backend == "llama_cpp":
        model_name = _resolve_model_name(settings)
        return LlamaCppLLMClient(
            base_url=settings.llm_server_url,
            model_name=model_name,
            timeout_seconds=settings.llm_timeout_seconds,
            max_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
            min_p=settings.llm_min_p,
            repeat_penalty=settings.llm_repeat_penalty,
        )
    if backend == "fake":
        return FakeLLMClient()
    raise ValueError(f"Unsupported llm_backend: {backend!r}")


def _resolve_model_name(settings: Settings) -> str:
    variant = settings.llm_model_variant.lower()
    return f"gemma-4-{variant}"
