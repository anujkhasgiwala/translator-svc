"""TTSEngine interface for text-to-audio synthesis."""

from abc import ABC, abstractmethod


class TTSEngine(ABC):
    @property
    @abstractmethod
    def supported_languages(self) -> set[str]:
        """ISO language codes this engine can synthesize."""

    @abstractmethod
    async def synthesize(self, text: str, lang: str) -> bytes:
        """Synthesize `text` in `lang` and return the audio bytes.

        Output format is impl-defined but consistent within one engine instance.
        """

    async def aclose(self) -> None:
        """Release any resources held by the engine. Default no-op."""
