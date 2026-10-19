"""LLMClient interface for audio-in / text-out translation models."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator


class LLMClient(ABC):
    @abstractmethod
    async def translate_audio(
        self,
        audio_bytes: bytes,
        audio_mime: str,
        prompt: str,
        system_prompt: str = "",
    ) -> str:
        """Run one audio-to-text inference pass.

        Returns the raw model output (containing both source transcript and target
        translation, to be split by `TranslationParser`). Returns an empty string
        when the model produced no translation (e.g. noise/silence audio), which
        the pipeline treats as a no-speech result rather than an error.
        """

    @abstractmethod
    def stream_translate_audio(
        self,
        audio_bytes: bytes,
        audio_mime: str,
        prompt: str,
        system_prompt: str = "",
    ) -> AsyncIterator[str]:
        """Run one audio-to-text inference pass, yielding output text deltas.

        Yields incremental chunks of the raw model output as they are generated,
        in order. Concatenating all yielded chunks reconstructs exactly what
        `translate_audio` would have returned. Yields nothing for a no-speech
        result (the pipeline treats an empty accumulation as no-speech rather
        than an error).
        """

    @abstractmethod
    async def health_check(self) -> bool:
        """Returns True if the upstream model server is reachable and ready."""

    async def aclose(self) -> None:
        """Release any resources held by the client. Default no-op."""
