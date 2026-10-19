"""SessionStore interface for rolling-context conversation state."""

from abc import ABC, abstractmethod
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass


@dataclass(frozen=True)
class Phrase:
    source_text: str
    target_text: str
    source_lang: str
    target_lang: str
    timestamp: float


class SessionStore(ABC):
    @abstractmethod
    def acquire_lock(self, session_id: str) -> AbstractAsyncContextManager[None]:
        """Per-session mutex. Returns an async context manager.

        Shape chosen so the in-memory impl (asyncio.Lock) and a future Redis impl
        (Redlock) both fit the same contract.
        """

    @abstractmethod
    async def get_context(
        self, session_id: str, source_lang: str, target_lang: str
    ) -> list[Phrase]:
        """Return up to N most-recent phrases for the (session, direction) pair."""

    @abstractmethod
    async def append(self, session_id: str, phrase: Phrase) -> None:
        """Append a phrase, evicting oldest if over capacity. Resets the session TTL."""
