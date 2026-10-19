"""In-memory SessionStore for single-replica deployments.

Per-direction deques (so EN->ES and ES->EN keep independent context), TTL
eviction, and per-session asyncio.Lock for ordering.
"""

import asyncio
from collections import deque
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from time import time

from .base import Phrase, SessionStore


class InMemorySessionStore(SessionStore):
    def __init__(
        self,
        max_phrases: int = 3,
        ttl_seconds: int = 1800,
    ) -> None:
        self._max_phrases = max_phrases
        self._ttl = ttl_seconds
        self._sessions: dict[str, dict[tuple[str, str], deque[Phrase]]] = {}
        self._last_seen: dict[str, float] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def acquire_lock(self, session_id: str) -> AbstractAsyncContextManager[None]:
        lock = self._locks.setdefault(session_id, asyncio.Lock())
        return self._lock_cm(lock)

    @asynccontextmanager
    async def _lock_cm(self, lock: asyncio.Lock):
        await lock.acquire()
        try:
            yield
        finally:
            lock.release()

    async def get_context(
        self, session_id: str, source_lang: str, target_lang: str
    ) -> list[Phrase]:
        self._evict_expired()
        buf = self._sessions.get(session_id, {}).get((source_lang, target_lang))
        return list(buf) if buf else []

    async def append(self, session_id: str, phrase: Phrase) -> None:
        self._evict_expired()
        key = (phrase.source_lang, phrase.target_lang)
        session = self._sessions.setdefault(session_id, {})
        buf = session.get(key)
        if buf is None:
            buf = deque(maxlen=self._max_phrases)
            session[key] = buf
        buf.append(phrase)
        self._last_seen[session_id] = time()

    def _evict_expired(self) -> None:
        now = time()
        expired = [
            sid for sid, last in self._last_seen.items() if now - last > self._ttl
        ]
        for sid in expired:
            self._sessions.pop(sid, None)
            self._last_seen.pop(sid, None)
            self._locks.pop(sid, None)
