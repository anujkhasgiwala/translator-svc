"""FastAPI Depends factories for the translator subpackage.

Each `get_*` reads a singleton attached to `app.state` by
`on_translator_startup`. `get_translation_service` is the per-request DI graph
that composes them into a `TranslationService`.
"""

from fastapi import Depends  # noqa: E402  (must come after typing import)
from fastapi import Request
from typing_extensions import Annotated

from translator_svc.config import Settings, get_settings

from .framing import FrameEncoder
from .llm.base import LLMClient
from .prompts import PromptBuilder, TranslationParser, TranslationValidator
from .service import TranslationService
from .sessions.base import SessionStore
from .tts.base import TTSEngine


def get_llm_client(request: Request) -> LLMClient:
    return request.app.state.llm_client


def get_tts_engine(request: Request) -> TTSEngine:
    return request.app.state.tts_engine


def get_session_store(request: Request) -> SessionStore:
    return request.app.state.session_store


def get_prompt_builder(request: Request) -> PromptBuilder:
    return request.app.state.prompt_builder


def get_translation_parser(request: Request) -> TranslationParser:
    return request.app.state.translation_parser


def get_translation_validator(request: Request) -> TranslationValidator:
    return request.app.state.translation_validator


def get_frame_encoder(request: Request) -> FrameEncoder:
    return request.app.state.frame_encoder


def get_translation_service(
    llm: Annotated[LLMClient, Depends(get_llm_client)],
    tts: Annotated[TTSEngine, Depends(get_tts_engine)],
    sessions: Annotated[SessionStore, Depends(get_session_store)],
    prompts: Annotated[PromptBuilder, Depends(get_prompt_builder)],
    parser: Annotated[TranslationParser, Depends(get_translation_parser)],
    validator: Annotated[TranslationValidator, Depends(get_translation_validator)],
    frames: Annotated[FrameEncoder, Depends(get_frame_encoder)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> TranslationService:
    return TranslationService(
        llm=llm,
        tts=tts,
        sessions=sessions,
        prompts=prompts,
        parser=parser,
        validator=validator,
        frames=frames,
        settings=settings,
    )
