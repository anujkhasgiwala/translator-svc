"""TranslationService — pipeline orchestrator for one phrase.

The only place that knows the full flow:
  lock -> context -> prompt -> LLM (streamed, accumulated) -> parse
        -> emit SOURCE + TARGET text frames
        -> validate
        -> launch per-sentence TTS tasks (concurrent) -> drain in order -> AUDIO
        -> append (only if valid)

The LLM output is streamed and accumulated; we parse the complete text with the
same last-labeled-line rule the non-streaming path used, so source/target
splitting is identical. TTS for each target sentence is launched as a concurrent
task and the resulting AUDIO frames are drained in sentence order — concurrency
overlaps synthesis compute but never reorders output (order matters: the audio
is conversational).

See DESIGN.md sections 7 and 9.3 for the data flow and cancellation invariants.
"""

import asyncio
import logging
import re
from collections.abc import AsyncIterator
from time import perf_counter, time

from translator_svc.config import Settings

from .framing import FrameEncoder, FrameType
from .llm.base import LLMClient
from .prompts import PromptBuilder, TranslationParser, TranslationValidator
from .sessions.base import Phrase, SessionStore
from .tts.base import TTSEngine

logger = logging.getLogger(__name__)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


class TranslationService:
    def __init__(
        self,
        llm: LLMClient,
        tts: TTSEngine,
        sessions: SessionStore,
        prompts: PromptBuilder,
        parser: TranslationParser,
        validator: TranslationValidator,
        frames: FrameEncoder,
        settings: Settings,
    ) -> None:
        self._llm = llm
        self._tts = tts
        self._sessions = sessions
        self._prompts = prompts
        self._parser = parser
        self._validator = validator
        self._frames = frames
        self._settings = settings

    async def translate_streaming(
        self,
        *,
        audio_bytes: bytes,
        audio_mime: str,
        source_lang: str,
        target_lang: str,
        session_id: str,
        seq: int,
    ) -> AsyncIterator[bytes]:
        pipeline_start = perf_counter()
        logger.info(
            "Pipeline start: session_id=%s seq=%s %s->%s audio_bytes=%d mime=%s",
            session_id,
            seq,
            source_lang,
            target_lang,
            len(audio_bytes),
            audio_mime,
        )
        async with self._sessions.acquire_lock(session_id):
            context = await self._sessions.get_context(
                session_id, source_lang, target_lang
            )
            system_prompt, user_prompt = self._prompts.build(
                context, source_lang, target_lang
            )
            logger.debug(
                "LLM input: session_id=%s seq=%s system_prompt_chars=%d "
                "user_prompt_chars=%d context_phrases=%d",
                session_id,
                seq,
                len(system_prompt),
                len(user_prompt),
                len(context),
            )

            # Stream the model output and accumulate it. Streaming lets the
            # response begin flushing as soon as content is parsed rather than
            # after the whole HTTP body is buffered; the accumulated text is
            # parsed with the same rule as the non-streaming path.
            llm_start = perf_counter()
            parts: list[str] = []
            first_delta_at: float | None = None
            async for delta in self._llm.stream_translate_audio(
                audio_bytes, audio_mime, user_prompt, system_prompt
            ):
                if first_delta_at is None:
                    first_delta_at = perf_counter()
                parts.append(delta)
            raw_text = "".join(parts)
            llm_elapsed_ms = (perf_counter() - llm_start) * 1000
            ttfb_ms = (
                (first_delta_at - llm_start) * 1000
                if first_delta_at is not None
                else None
            )
            logger.info(
                "LLM done: session_id=%s seq=%s elapsed_ms=%.1f ttfb_ms=%s "
                "deltas=%d output_chars=%d",
                session_id,
                seq,
                llm_elapsed_ms,
                f"{ttfb_ms:.1f}" if ttfb_ms is not None else "n/a",
                len(parts),
                len(raw_text),
            )

            # No content from the model (e.g. noise/silence audio) is a valid
            # no-speech outcome, not an error. Emit empty text frames and stop;
            # the UI shows nothing and the phrase is not added to context.
            if not raw_text.strip():
                logger.info(
                    "No translation produced; treating as no-speech",
                    extra={"session_id": session_id, "seq": seq},
                )
                yield self._frames.encode(FrameType.SOURCE, b"")
                yield self._frames.encode(FrameType.TARGET, b"")
                return

            source_text, target_text = self._parser.parse(raw_text)
            logger.info(
                "Parsed translation: session_id=%s seq=%s "
                "source_chars=%d target_chars=%d",
                session_id,
                seq,
                len(source_text),
                len(target_text),
            )
            # Full translation text is user speech content; keep it at DEBUG so
            # it isn't shipped to shared log aggregators at the default level.
            logger.debug("Source text: %s", source_text)
            logger.debug("Target text: %s", target_text)

            yield self._frames.encode(FrameType.SOURCE, source_text.encode("utf-8"))
            yield self._frames.encode(FrameType.TARGET, target_text.encode("utf-8"))

            valid = self._validator.is_valid(source_text, target_text)

            # Pipeline TTS: launch synthesis for every sentence up front so they
            # run concurrently (Kokoro runs each on a thread), then DRAIN the
            # tasks in sentence order — await task i, emit its AUDIO frame, then
            # task i+1. Awaiting by index guarantees audio is emitted in sentence
            # order regardless of which synthesis finishes first; concurrency
            # overlaps compute only, never reorders output. A failed sentence is
            # logged and skipped so one bad sentence doesn't drop the rest.
            sentences = self._split_sentences(target_text)
            tts_start = perf_counter()
            logger.info(
                "TTS start: session_id=%s seq=%s sentences=%d lang=%s",
                session_id,
                seq,
                len(sentences),
                target_lang,
            )
            tasks = [
                asyncio.ensure_future(self._tts.synthesize(s, target_lang))
                for s in sentences
            ]
            emitted = 0
            total_audio_bytes = 0
            try:
                for idx, (sentence, task) in enumerate(zip(sentences, tasks)):
                    sentence_start = perf_counter()
                    try:
                        audio_chunk = await task
                    except Exception:
                        logger.warning(
                            "TTS failed for sentence: session_id=%s seq=%s "
                            "index=%d text_chars=%d",
                            session_id,
                            seq,
                            idx,
                            len(sentence),
                        )
                        continue
                    sentence_ms = (perf_counter() - sentence_start) * 1000
                    total_audio_bytes += len(audio_chunk)
                    emitted += 1
                    logger.debug(
                        "TTS sentence done: session_id=%s seq=%s index=%d "
                        "text_chars=%d audio_bytes=%d drain_wait_ms=%.1f",
                        session_id,
                        seq,
                        idx,
                        len(sentence),
                        len(audio_chunk),
                        sentence_ms,
                    )
                    yield self._frames.encode(FrameType.AUDIO, audio_chunk)
            finally:
                # If we exit the drain early (client disconnect -> generator
                # cancelled), don't leave synthesis tasks running detached.
                for task in tasks:
                    if not task.done():
                        task.cancel()
            tts_elapsed_ms = (perf_counter() - tts_start) * 1000
            logger.info(
                "TTS done: session_id=%s seq=%s elapsed_ms=%.1f "
                "sentences_emitted=%d/%d total_audio_bytes=%d",
                session_id,
                seq,
                tts_elapsed_ms,
                emitted,
                len(sentences),
                total_audio_bytes,
            )

            if valid:
                await self._sessions.append(
                    session_id,
                    Phrase(
                        source_text=source_text,
                        target_text=target_text,
                        source_lang=source_lang,
                        target_lang=target_lang,
                        timestamp=time(),
                    ),
                )
            else:
                logger.info(
                    "Translation failed validation; not appending to context: "
                    "session_id=%s seq=%s source_chars=%d target_chars=%d",
                    session_id,
                    seq,
                    len(source_text),
                    len(target_text),
                )
                # Text is user speech content; DEBUG only (see parse logging).
                logger.debug(
                    "Failed-validation text: source=%s target=%s",
                    source_text,
                    target_text,
                )

            logger.info(
                "Pipeline done: session_id=%s seq=%s total_ms=%.1f "
                "llm_ms=%.1f tts_ms=%.1f valid=%s",
                session_id,
                seq,
                (perf_counter() - pipeline_start) * 1000,
                llm_elapsed_ms,
                tts_elapsed_ms,
                valid,
            )

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        parts = [p.strip() for p in _SENTENCE_SPLIT_RE.split(text.strip())]
        return [p for p in parts if p]
