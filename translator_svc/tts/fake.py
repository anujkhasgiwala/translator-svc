"""In-memory fake TTSEngine for tests and end-to-end scaffolding."""

from .base import TTSEngine


class FakeTTSEngine(TTSEngine):
    def __init__(self, supported: set[str] | None = None) -> None:
        self._supported = supported or {"English", "Spanish"}

    @property
    def supported_languages(self) -> set[str]:
        return self._supported

    async def synthesize(self, text: str, lang: str) -> bytes:
        return f"FAKE_AUDIO[{lang}]:{text}".encode("utf-8")
