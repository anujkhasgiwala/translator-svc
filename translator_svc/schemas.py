"""Request/response value objects for the translator endpoint.

Header and form parsing is done in `router.py` via FastAPI's `Header(...)` /
`Form(...)` dependencies. The structs here capture the validated, normalized
shape that flows into `TranslationService`.
"""

from dataclasses import dataclass
from uuid import UUID


@dataclass(frozen=True)
class TranslateRequest:
    audio_bytes: bytes
    audio_mime: str
    source_lang: str
    target_lang: str
    session_id: UUID
    seq: int


@dataclass(frozen=True)
class ErrorPayload:
    code: str
    message: str
