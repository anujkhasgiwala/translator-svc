"""Unit tests for LlamaCppLLMClient response handling."""

import httpx
import pytest

from translator_svc.errors import LLMUnavailableError
from translator_svc.llm.llama_cpp import LlamaCppLLMClient


def _make_client(handler: httpx.MockTransport) -> LlamaCppLLMClient:
    client = LlamaCppLLMClient(
        base_url="http://llama",
        model_name="gemma-4-e4b",
    )
    # Swap in a mock transport so no real network call happens.
    client._client = httpx.AsyncClient(base_url="http://llama", transport=handler)
    return client


def _chat_response(message: dict, finish_reason: str = "stop") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [
                {"finish_reason": finish_reason, "index": 0, "message": message}
            ]
        },
    )


@pytest.mark.anyio
async def test_returns_content_when_present() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _chat_response({"role": "assistant", "content": "Spanish: hola"})

    client = _make_client(httpx.MockTransport(handler))
    result = await client.translate_audio(b"\x00", "audio/wav", "prompt")
    assert result == "Spanish: hola"


@pytest.mark.anyio
async def test_system_prompt_sent_as_system_message() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        captured.update(json.loads(request.content))
        return _chat_response({"role": "assistant", "content": "Spanish: hola"})

    client = _make_client(httpx.MockTransport(handler))
    await client.translate_audio(b"\x00", "audio/wav", "user text", "system text")

    messages = captured["messages"]
    assert messages[0] == {"role": "system", "content": "system text"}
    assert messages[1]["role"] == "user"
    # The user message keeps the text + audio content blocks.
    assert messages[1]["content"][0] == {"type": "text", "text": "user text"}
    assert messages[1]["content"][1]["type"] == "input_audio"


@pytest.mark.anyio
async def test_no_system_message_when_system_prompt_empty() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        captured.update(json.loads(request.content))
        return _chat_response({"role": "assistant", "content": "Spanish: hola"})

    client = _make_client(httpx.MockTransport(handler))
    await client.translate_audio(b"\x00", "audio/wav", "user text")

    assert all(m["role"] != "system" for m in captured["messages"])


@pytest.mark.anyio
async def test_empty_content_returns_empty_string_not_error() -> None:
    """Reasoning runaway / noise audio: 200 with empty content is no-speech."""

    def handler(request: httpx.Request) -> httpx.Response:
        return _chat_response(
            {"role": "assistant", "content": "", "reasoning_content": "lots of think"}
        )

    client = _make_client(httpx.MockTransport(handler))
    result = await client.translate_audio(b"\x00", "audio/wav", "prompt")
    assert result == ""


@pytest.mark.anyio
async def test_sends_tuned_sampling_params() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        captured.update(json.loads(request.content))
        return _chat_response({"role": "assistant", "content": "Spanish: hola"})

    client = _make_client(httpx.MockTransport(handler))
    await client.translate_audio(b"\x00", "audio/wav", "prompt")
    # Sampling tuned for Gemma 4 reasoning: min_p nucleus (adapts to model
    # confidence) plus a light repeat_penalty, with Thinking Mode left on.
    assert captured["min_p"] == 0.05
    assert captured["repeat_penalty"] == 1.05
    assert "top_p" not in captured
    assert "top_k" not in captured


@pytest.mark.anyio
async def test_server_5xx_raises_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="overloaded")

    client = _make_client(httpx.MockTransport(handler))
    with pytest.raises(LLMUnavailableError):
        await client.translate_audio(b"\x00", "audio/wav", "prompt")


def _sse_body(deltas: list[str]) -> bytes:
    """Build an OpenAI-style SSE chat-completion stream body."""
    import json

    lines: list[str] = []
    for delta in deltas:
        chunk = {"choices": [{"index": 0, "delta": {"content": delta}}]}
        lines.append(f"data: {json.dumps(chunk)}")
    # A final chunk with finish_reason and no content, then the DONE sentinel.
    lines.append(
        f"data: {json.dumps({'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}"
    )
    lines.append("data: [DONE]")
    return ("\n\n".join(lines) + "\n\n").encode("utf-8")


@pytest.mark.anyio
async def test_stream_yields_content_deltas_in_order() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_sse_body(["English: ", "hello\n", "Spanish: ", "hola"]),
        )

    client = _make_client(httpx.MockTransport(handler))
    chunks = [c async for c in client.stream_translate_audio(b"\x00", "audio/wav", "p")]
    assert chunks == ["English: ", "hello\n", "Spanish: ", "hola"]
    assert "".join(chunks) == "English: hello\nSpanish: hola"


@pytest.mark.anyio
async def test_stream_sets_stream_true_and_keeps_sampling() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_sse_body(["Spanish: hola"]),
        )

    client = _make_client(httpx.MockTransport(handler))
    async for _ in client.stream_translate_audio(b"\x00", "audio/wav", "p"):
        pass

    assert captured["stream"] is True
    assert captured["min_p"] == 0.05
    assert captured["repeat_penalty"] == 1.05


@pytest.mark.anyio
async def test_stream_skips_empty_deltas() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_sse_body(["", "hola", ""]),
        )

    client = _make_client(httpx.MockTransport(handler))
    chunks = [c async for c in client.stream_translate_audio(b"\x00", "audio/wav", "p")]
    assert chunks == ["hola"]


@pytest.mark.anyio
async def test_stream_server_5xx_raises_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="overloaded")

    client = _make_client(httpx.MockTransport(handler))
    with pytest.raises(LLMUnavailableError):
        async for _ in client.stream_translate_audio(b"\x00", "audio/wav", "p"):
            pass
