"""SessionStore factory."""

from translator_svc.config import Settings

from .base import Phrase, SessionStore
from .memory import InMemorySessionStore
from .redis import RedisSessionStore

__all__ = ["Phrase", "SessionStore", "make_session_store"]


def make_session_store(settings: Settings) -> SessionStore:
    backend = settings.session_store_backend
    if backend == "memory":
        return InMemorySessionStore(
            max_phrases=settings.session_max_phrases,
            ttl_seconds=settings.session_ttl_seconds,
        )
    if backend == "redis":
        return RedisSessionStore(
            redis_url=settings.redis_url,
            max_phrases=settings.session_max_phrases,
            ttl_seconds=settings.session_ttl_seconds,
        )
    raise ValueError(f"Unsupported session_store_backend: {backend!r}")
