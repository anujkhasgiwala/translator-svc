"""Translator startup/shutdown hooks.

Called from the app lifespan in `main.py`. Builds singletons via
factories, attaches them to `app.state`, and registers them with the
`AsyncExitStack` so cleanup happens automatically on shutdown.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .framing import FrameEncoder
from .llm import make_llm_client
from .llm.base import LLMClient
from .prompts import PromptBuilder, TranslationParser, TranslationValidator
from .sessions import make_session_store
from .tts import make_tts_engine
from .tts.base import TTSEngine

logger = logging.getLogger(__name__)


async def on_translator_startup(app: FastAPI) -> None:
    settings = app.state.config

    llm = make_llm_client(settings)
    if not await llm.health_check():
        await llm.aclose()
        raise RuntimeError(
            f"LLM unreachable at startup: {settings.llm_server_url} "
            f"(backend={settings.llm_backend})"
        )
    app.state.llm_client = await app.state.async_exit_stack.enter_async_context(
        _client_lifespan(llm)
    )
    logger.info(
        "LLM client ready",
        extra={
            "backend": settings.llm_backend,
            "url": settings.llm_server_url,
            "model_variant": settings.llm_model_variant,
        },
    )

    tts = make_tts_engine(settings)
    app.state.tts_engine = await app.state.async_exit_stack.enter_async_context(
        _engine_lifespan(tts)
    )
    logger.info(
        "TTS engine ready",
        extra={
            "backend": settings.tts_backend,
            "supported_languages": sorted(tts.supported_languages),
        },
    )

    app.state.session_store = make_session_store(settings)
    app.state.prompt_builder = PromptBuilder(
        prompt_truncate_chars=settings.session_prompt_truncate_chars,
    )
    app.state.translation_parser = TranslationParser()
    app.state.translation_validator = TranslationValidator(
        min_ratio=settings.validation_min_ratio,
        max_ratio=settings.validation_max_ratio,
        refusal_patterns=settings.validation_refusal_patterns,
    )
    app.state.frame_encoder = FrameEncoder()

    logger.info(
        "Translator subsystem ready",
        extra={
            "session_store": settings.session_store_backend,
            "supported_languages": sorted(settings.supported_languages),
        },
    )


@asynccontextmanager
async def _client_lifespan(client: LLMClient):
    try:
        yield client
    finally:
        await client.aclose()


@asynccontextmanager
async def _engine_lifespan(engine: TTSEngine):
    try:
        yield engine
    finally:
        await engine.aclose()
