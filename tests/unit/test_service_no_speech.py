"""Service-level tests for the no-speech (empty LLM output) path."""

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
from translator_svc.tts.fake import FakeTTSEngine


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


def _make_service(responder) -> TranslationService:
    settings = Settings()
    return TranslationService(
        llm=FakeLLMClient(responder=responder),
        tts=FakeTTSEngine(),
        sessions=InMemorySessionStore(),
        prompts=PromptBuilder(),
        parser=TranslationParser(),
        validator=TranslationValidator(),
        frames=FrameEncoder(),
        settings=settings,
    )


@pytest.mark.anyio
async def test_empty_llm_output_emits_empty_frames_no_error() -> None:
    service = _make_service(lambda *_: "")  # model produced nothing

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

    frames = _decode_frames(out)
    # Only empty SOURCE and TARGET frames; no ERROR, no AUDIO.
    assert frames == [(FrameType.SOURCE, b""), (FrameType.TARGET, b"")]
    assert all(f[0] != FrameType.ERROR for f in frames)


@pytest.mark.anyio
async def test_no_speech_not_added_to_context() -> None:
    service = _make_service(lambda *_: "")
    session_id = "22222222-2222-2222-2222-222222222222"

    async for _ in service.translate_streaming(
        audio_bytes=b"\x00",
        audio_mime="audio/wav",
        source_lang="English",
        target_lang="Spanish",
        session_id=session_id,
        seq=1,
    ):
        pass

    context = await service._sessions.get_context(session_id, "English", "Spanish")
    assert list(context) == []
