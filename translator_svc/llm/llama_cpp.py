"""LlamaCppLLMClient — talks to a llama-server over its OpenAI-compatible API.

Sends one phrase as a chat-completions request with an `input_audio` content
block; the model is expected to return both the source transcript and the
target translation in its assistant message (parsed downstream by
TranslationParser).
"""

import base64
import json
import logging
from collections.abc import AsyncIterator
from time import perf_counter

import httpx

from ..errors import LLMUnavailableError
from .base import LLMClient

logger = logging.getLogger(__name__)


_MIME_TO_FORMAT = {
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/wave": "wav",
    "audio/webm": "webm",
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
    "audio/ogg": "ogg",
    "audio/flac": "flac",
}


class LlamaCppLLMClient(LLMClient):
    def __init__(
        self,
        base_url: str,
        model_name: str,
        timeout_seconds: float = 30.0,
        max_tokens: int = 256,
        temperature: float = 0.7,
        min_p: float = 0.05,
        repeat_penalty: float = 1.05,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model_name = model_name
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._min_p = min_p
        self._repeat_penalty = repeat_penalty
        self._client = httpx.AsyncClient(
            base_url=self._base_url, timeout=timeout_seconds
        )

    def _build_payload(
        self,
        audio_bytes: bytes,
        audio_mime: str,
        prompt: str,
        system_prompt: str,
        *,
        stream: bool,
    ) -> dict:
        audio_format = _MIME_TO_FORMAT.get(audio_mime.lower(), "wav")
        audio_b64 = base64.b64encode(audio_bytes).decode("ascii")

        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": audio_b64,
                            "format": audio_format,
                        },
                    },
                ],
            }
        )

        return {
            "model": self._model_name,
            "messages": messages,
            "stream": stream,
            "max_tokens": self._max_tokens,
            # Sampling for Gemma 4 with reasoning OFF: temperature keeps the
            # direct answer natural/idiomatic, min_p keeps the nucleus narrow
            # enough to exclude low-probability phonetic siblings (e.g.
            # "cono serde"), and repeat_penalty discourages repetition loops.
            "temperature": self._temperature,
            "min_p": self._min_p,
            "repeat_penalty": self._repeat_penalty,
            # Disable Gemma's reasoning trace at the chat-template level. Without
            # this the model intermittently emits a multi-hundred-token
            # "Thinking Process" trace that exhausts max_tokens before producing
            # any answer (finish_reason=length, content empty -> handled as a
            # no-speech result downstream). The llama-server launch flag
            # (--reasoning off) does the same thing, but setting it per-request
            # makes the service correct regardless of how the sidecar was
            # started. Verified: with this set, runs return finish=stop with no
            # reasoning_content; without it, ~1 in 5 runs went empty.
            "chat_template_kwargs": {"enable_thinking": False},
        }

    async def translate_audio(
        self,
        audio_bytes: bytes,
        audio_mime: str,
        prompt: str,
        system_prompt: str = "",
    ) -> str:
        payload = self._build_payload(
            audio_bytes, audio_mime, prompt, system_prompt, stream=False
        )
        logger.info(
            "LLM request (non-stream): model=%s audio_bytes=%d prompt_chars=%d "
            "system_prompt_chars=%d max_tokens=%d",
            self._model_name,
            len(audio_bytes),
            len(prompt),
            len(system_prompt),
            self._max_tokens,
        )

        start = perf_counter()
        try:
            resp = await self._client.post("/v1/chat/completions", json=payload)
        except httpx.HTTPError as e:
            raise LLMUnavailableError(f"llama-server unreachable: {e}") from e

        if resp.status_code >= 500:
            raise LLMUnavailableError(
                f"llama-server returned {resp.status_code}: {resp.text[:200]}"
            )
        if resp.status_code >= 400:
            raise LLMUnavailableError(
                f"llama-server rejected request ({resp.status_code}): {resp.text[:200]}"
            )

        elapsed_ms = (perf_counter() - start) * 1000
        body = resp.json()
        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise LLMUnavailableError(
                f"Unexpected llama-server response shape: {body!r}"
            ) from e

        usage = body.get("usage") or {}
        logger.info(
            "LLM response (non-stream): elapsed_ms=%.1f finish_reason=%s "
            "output_chars=%d prompt_tokens=%s completion_tokens=%s",
            elapsed_ms,
            self._finish_reason(body),
            len(content) if isinstance(content, str) else 0,
            usage.get("prompt_tokens", "n/a"),
            usage.get("completion_tokens", "n/a"),
        )

        # Empty content on a 200 is not a server failure: the model produced no
        # translation (e.g. noise/silence audio, or it exhausted the budget
        # reasoning). Return "" so the pipeline treats it as a no-speech result
        # rather than raising llm_unavailable.
        if not isinstance(content, str) or not content.strip():
            finish_reason = self._finish_reason(body)
            logger.info(
                "llama-server returned empty content (finish_reason=%s); "
                "treating as no-speech",
                finish_reason,
            )
            return ""
        return content

    async def stream_translate_audio(
        self,
        audio_bytes: bytes,
        audio_mime: str,
        prompt: str,
        system_prompt: str = "",
    ) -> AsyncIterator[str]:
        payload = self._build_payload(
            audio_bytes, audio_mime, prompt, system_prompt, stream=True
        )
        logger.info(
            "LLM request (stream): model=%s audio_bytes=%d prompt_chars=%d "
            "system_prompt_chars=%d max_tokens=%d",
            self._model_name,
            len(audio_bytes),
            len(prompt),
            len(system_prompt),
            self._max_tokens,
        )

        start = perf_counter()
        first_delta_at: float | None = None
        deltas = 0
        output_chars = 0
        try:
            async with self._client.stream(
                "POST", "/v1/chat/completions", json=payload
            ) as resp:
                if resp.status_code >= 400:
                    # Body must be read before the response is closed.
                    body_text = (await resp.aread()).decode("utf-8", "replace")
                    raise LLMUnavailableError(
                        f"llama-server rejected stream ({resp.status_code}): "
                        f"{body_text[:200]}"
                    )
                async for line in resp.aiter_lines():
                    delta = self._parse_sse_line(line)
                    if delta:
                        if first_delta_at is None:
                            first_delta_at = perf_counter()
                        deltas += 1
                        output_chars += len(delta)
                        yield delta
        except httpx.HTTPError as e:
            raise LLMUnavailableError(f"llama-server unreachable: {e}") from e
        finally:
            elapsed_ms = (perf_counter() - start) * 1000
            ttfb_ms = (
                (first_delta_at - start) * 1000 if first_delta_at is not None else None
            )
            logger.info(
                "LLM response (stream): elapsed_ms=%.1f ttfb_ms=%s deltas=%d "
                "output_chars=%d",
                elapsed_ms,
                f"{ttfb_ms:.1f}" if ttfb_ms is not None else "n/a",
                deltas,
                output_chars,
            )

    @staticmethod
    def _parse_sse_line(line: str) -> str:
        """Extract the content delta from one SSE line, or "" if none.

        Lines are OpenAI-style ``data: {json}``; ``data: [DONE]`` terminates the
        stream; blank lines and keep-alives are ignored.
        """
        line = line.strip()
        if not line.startswith("data:"):
            return ""
        data = line[len("data:") :].strip()
        if not data or data == "[DONE]":
            return ""
        try:
            chunk = json.loads(data)
            content = chunk["choices"][0]["delta"].get("content")
        except (KeyError, IndexError, TypeError, ValueError):
            return ""
        return content if isinstance(content, str) else ""

    @staticmethod
    def _finish_reason(body: dict) -> str | None:
        try:
            return body["choices"][0].get("finish_reason")
        except (KeyError, IndexError, TypeError):
            return None

    async def health_check(self) -> bool:
        try:
            resp = await self._client.get("/health")
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    async def aclose(self) -> None:
        await self._client.aclose()
