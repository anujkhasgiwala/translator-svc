"""Service-level tests for the streaming LLM + pipelined TTS path.

These exercise translate_streaming consuming the LLM via stream_translate_audio
(the FakeLLMClient streams its canned response in chunks), and assert frame
ordering, the no-speech path, and the validate-then-append invariant.
"""

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


def _make_service(responder, tts=None) -> TranslationService:
    settings = Settings()
    return TranslationService(
        llm=FakeLLMClient(responder=responder, stream_chunk_size=3),
        tts=tts or FakeTTSEngine(),
        sessions=InMemorySessionStore(),
        prompts=PromptBuilder(),
        parser=TranslationParser(),
        validator=TranslationValidator(),
        frames=FrameEncoder(),
        settings=settings,
    )


async def _drain(service, **kwargs) -> list[tuple[int, bytes]]:
    out = b"".join([frame async for frame in service.translate_streaming(**kwargs)])
    return _decode_frames(out)


_KW = dict(
    audio_bytes=b"\x00",
    audio_mime="audio/wav",
    source_lang="English",
    target_lang="Spanish",
    seq=1,
)


@pytest.mark.anyio
async def test_streaming_emits_source_then_target_then_audio() -> None:
    service = _make_service(lambda *_: "English: hello\nSpanish: hola.")
    frames = await _drain(
        service, session_id="11111111-1111-1111-1111-111111111111", **_KW
    )

    types = [t for t, _ in frames]
    assert types[0] == FrameType.SOURCE
    assert types[1] == FrameType.TARGET
    assert all(t == FrameType.AUDIO for t in types[2:])

    by_type = {t: p for t, p in frames}
    assert by_type[FrameType.SOURCE] == b"hello"
    assert by_type[FrameType.TARGET] == b"hola."


@pytest.mark.anyio
async def test_streaming_multi_sentence_audio_in_order() -> None:
    service = _make_service(lambda *_: "English: a b c\nSpanish: uno. dos. tres.")
    frames = await _drain(
        service, session_id="22222222-2222-2222-2222-222222222222", **_KW
    )

    audio = [p for t, p in frames if t == FrameType.AUDIO]
    assert audio == [
        b"FAKE_AUDIO[Spanish]:uno.",
        b"FAKE_AUDIO[Spanish]:dos.",
        b"FAKE_AUDIO[Spanish]:tres.",
    ]


@pytest.mark.anyio
async def test_streaming_no_speech_emits_empty_frames_only() -> None:
    service = _make_service(lambda *_: "")
    frames = await _drain(
        service, session_id="33333333-3333-3333-3333-333333333333", **_KW
    )
    assert frames == [(FrameType.SOURCE, b""), (FrameType.TARGET, b"")]


@pytest.mark.anyio
async def test_streaming_valid_translation_appended_to_context() -> None:
    service = _make_service(lambda *_: "English: hello\nSpanish: hola.")
    session_id = "44444444-4444-4444-4444-444444444444"
    await _drain(service, session_id=session_id, **_KW)

    context = await service._sessions.get_context(session_id, "English", "Spanish")
    phrases = list(context)
    assert len(phrases) == 1
    assert phrases[0].target_text == "hola."
