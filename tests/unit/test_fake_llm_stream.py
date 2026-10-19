"""Tests for FakeLLMClient streaming: chunks concatenate to the full response."""

import pytest

from translator_svc.llm.fake import FakeLLMClient


@pytest.mark.anyio
async def test_stream_chunks_reconstruct_full_response() -> None:
    client = FakeLLMClient(
        responder=lambda *_: "English: hello\nSpanish: hola",
        stream_chunk_size=4,
    )

    chunks = [c async for c in client.stream_translate_audio(b"\x00", "audio/wav", "p")]

    assert len(chunks) > 1  # actually chunked
    assert "".join(chunks) == "English: hello\nSpanish: hola"


@pytest.mark.anyio
async def test_stream_empty_response_yields_nothing() -> None:
    client = FakeLLMClient(responder=lambda *_: "")

    chunks = [c async for c in client.stream_translate_audio(b"\x00", "audio/wav", "p")]

    assert chunks == []
