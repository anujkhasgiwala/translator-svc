"""Redis-backed SessionStore stub.

Contract is fully defined for distributed deployments (multi-pod k8s). The
implementation is intentionally a NotImplementedError stub; flipping
`SESSION_STORE_BACKEND` from "memory" to "redis" requires implementing this
class without touching pipeline code.

Planned implementation outline (do not implement until needed):
- Per-session lock: Redlock (e.g. redis-py-lock) over `lock:<session_id>`.
- Context buffer: Redis list at `ctx:<session_id>:<src>:<tgt>`, JSON-serialized
  Phrase entries, LTRIM to max_phrases, EXPIRE on every write for TTL.
- Connection: shared redis.asyncio.Redis client.
"""

from contextlib import AbstractAsyncContextManager

from .base import Phrase, SessionStore


class RedisSessionStore(SessionStore):
    def __init__(self, redis_url: str, max_phrases: int = 3, ttl_seconds: int = 1800):
        self._redis_url = redis_url
        self._max_phrases = max_phrases
        self._ttl = ttl_seconds

    def acquire_lock(self, session_id: str) -> AbstractAsyncContextManager[None]:
        raise NotImplementedError("RedisSessionStore is not yet implemented")

    async def get_context(
        self, session_id: str, source_lang: str, target_lang: str
    ) -> list[Phrase]:
        raise NotImplementedError("RedisSessionStore is not yet implemented")

    async def append(self, session_id: str, phrase: Phrase) -> None:
        raise NotImplementedError("RedisSessionStore is not yet implemented")
