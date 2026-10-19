"""In-memory fake LLMClient for tests and end-to-end scaffolding."""

from collections.abc import AsyncIterator, Callable

from .base import LLMClient


class FakeLLMClient(LLMClient):
    def __init__(
        self,
        responder: Callable[[bytes, str, str], str] | None = None,
        healthy: bool = True,
        stream_chunk_size: int = 8,
    ) -> None:
        self._responder = responder or self._default_responder
        self._healthy = healthy
        self._stream_chunk_size = stream_chunk_size

    async def translate_audio(
        self,
        audio_bytes: bytes,
        audio_mime: str,
        prompt: str,
        system_prompt: str = "",
    ) -> str:
        return self._responder(audio_bytes, audio_mime, prompt)

    async def stream_translate_audio(
        self,
        audio_bytes: bytes,
        audio_mime: str,
        prompt: str,
        system_prompt: str = "",
    ) -> AsyncIterator[str]:
        # Emit the canned response in fixed-size chunks so streaming consumers
        # see multiple deltas. Concatenating the chunks reproduces the full text.
        text = self._responder(audio_bytes, audio_mime, prompt)
        for i in range(0, len(text), self._stream_chunk_size):
            yield text[i : i + self._stream_chunk_size]

    async def health_check(self) -> bool:
        return self._healthy

    @staticmethod
    def _default_responder(audio_bytes: bytes, audio_mime: str, prompt: str) -> str:
        return "English: hello\nSpanish: hola"
