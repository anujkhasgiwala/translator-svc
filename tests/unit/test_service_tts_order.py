"""Service-level tests for parallel TTS: audio frame ORDER must match the
sentence order of the translation, regardless of per-sentence synthesis time.

Order matters because the audio is conversational — sentences played out of
order would be unintelligible.
"""

import asyncio

import pytest

from translator_svc.config import Settings
from translator_svc.framing import FrameEncoder, FrameType
from translator_svc.llm.fake import FakeLLMClient
from translator_svc.prompts import (
    PromptBuilder,
    TranslationParser,
    TranslationValidator,
)
from translator_svc.service import TranslationService
from translator_svc.sessions.memory import InMemorySessionStore
from translator_svc.tts.base import TTSEngine


def _decode_frames(blob: bytes) -> list[tuple[int, bytes]]:
    frames: list[tuple[int, bytes]] = []
    i = 0
    while i < len(blob):
        ftype = blob[i]
        length = int.from_bytes(blob[i + 1 : i + 5], "big")
        payload = blob[i + 5 : i + 5 + length]
        frames.append((ftype, payload))
        i += 5 + length
    return frames


class _ReverseDelayTTS(TTSEngine):
    """Synthesizes earlier sentences SLOWER than later ones.

    If the service awaited sentences serially or yielded by completion order,
    audio frames would come out reordered. Correct (gather + positional emit)
    behavior keeps them in sentence order.
    """

    def __init__(self) -> None:
        self._call_index = 0

    @property
    def supported_languages(self) -> set[str]:
        return {"English", "Spanish"}

    async def synthesize(self, text: str, lang: str) -> bytes:
        # First sentence sleeps longest so it would finish last.
        index = self._call_index
        self._call_index += 1
        await asyncio.sleep(0.03 * (3 - index))
        return f"AUDIO:{text}".encode("utf-8")


class _FailSecondTTS(TTSEngine):
    """Fails synthesis for the second sentence only."""

    def __init__(self) -> None:
        self._call_index = 0

    @property
    def supported_languages(self) -> set[str]:
        return {"English", "Spanish"}

    async def synthesize(self, text: str, lang: str) -> bytes:
        index = self._call_index
        self._call_index += 1
        if index == 1:
            raise RuntimeError("boom")
        return f"AUDIO:{text}".encode("utf-8")


def _make_service(responder, tts: TTSEngine) -> TranslationService:
    settings = Settings()
    return TranslationService(
        llm=FakeLLMClient(responder=responder),
        tts=tts,
        sessions=InMemorySessionStore(),
        prompts=PromptBuilder(),
        parser=TranslationParser(),
        validator=TranslationValidator(),
        frames=FrameEncoder(),
        settings=settings,
    )


_THREE_SENTENCES = "English: one. two. three.\nSpanish: uno. dos. tres."


@pytest.mark.anyio
async def test_audio_frames_emitted_in_sentence_order_despite_reverse_timing() -> None:
    service = _make_service(lambda *_: _THREE_SENTENCES, _ReverseDelayTTS())

    out = b"".join(
        [
            frame
            async for frame in service.translate_streaming(
                audio_bytes=b"\x00",
                audio_mime="audio/wav",
                source_lang="English",
                target_lang="Spanish",
                session_id="11111111-1111-1111-1111-111111111111",
                seq=1,
            )
        ]
    )

    audio = [p for t, p in _decode_frames(out) if t == FrameType.AUDIO]
    assert audio == [b"AUDIO:uno.", b"AUDIO:dos.", b"AUDIO:tres."]


@pytest.mark.anyio
async def test_failed_sentence_skipped_others_keep_order() -> None:
    service = _make_service(lambda *_: _THREE_SENTENCES, _FailSecondTTS())

    out = b"".join(
        [
            frame
            async for frame in service.translate_streaming(
                audio_bytes=b"\x00",
                audio_mime="audio/wav",
                source_lang="English",
                target_lang="Spanish",
                session_id="22222222-2222-2222-2222-222222222222",
                seq=1,
            )
        ]
    )

    audio = [p for t, p in _decode_frames(out) if t == FrameType.AUDIO]
    # Second sentence failed -> skipped; first and third keep their order.
    assert audio == [b"AUDIO:uno.", b"AUDIO:tres."]
