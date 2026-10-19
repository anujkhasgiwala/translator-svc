"""KokoroLocalTTSEngine — in-process Kokoro 82M TTS.

One KPipeline is loaded per language at construction time (eager load per the
service's startup eagerness policy). Synthesis runs on a thread pool because
Kokoro is synchronous CPU/GPU work and would otherwise block the asyncio
event loop.

Output is encoded as 24 kHz, 16-bit mono PCM WAV bytes.
"""

import asyncio
import io
import logging
from collections.abc import Iterable
from time import perf_counter

import numpy as np
import soundfile as sf
from kokoro import KPipeline

from ..errors import TTSError
from .base import TTSEngine

logger = logging.getLogger(__name__)


# Map service-side language names -> Kokoro lang_code + default voice.
# Add languages here as we expand supported_languages in settings.
#
# Each language loads ONE default voice. Alternate voices Kokoro ships for a
# language can be selected at construction time via `voice_overrides` without
# touching this table:
#   Spanish: ef_dora (default), em_alex
#   Hindi:   hf_alpha (default), hm_omega
#   Italian: if_sara (default), im_nicola
_DEFAULT_VOICES: dict[str, tuple[str, str]] = {
    "English": ("a", "af_heart"),
    "Spanish": ("e", "ef_dora"),
    "Hindi": ("h", "hf_alpha"),
    "French": ("f", "ff_siwis"),
    "Italian": ("i", "if_sara"),
}

_SAMPLE_RATE = 24_000


class KokoroLocalTTSEngine(TTSEngine):
    def __init__(
        self,
        model_path: str | None = None,
        languages: Iterable[str] = ("English", "Spanish"),
        voice_overrides: dict[str, str] | None = None,
    ) -> None:
        # model_path is accepted for symmetry with settings but Kokoro pulls
        # weights from the HuggingFace cache; the field is unused for now.
        self._model_path = model_path
        self._pipelines: dict[str, KPipeline] = {}
        self._voices: dict[str, str] = {}

        overrides = voice_overrides or {}
        for lang in languages:
            if lang not in _DEFAULT_VOICES:
                logger.warning("Skipping unsupported Kokoro language: %s", lang)
                continue
            kokoro_code, default_voice = _DEFAULT_VOICES[lang]
            voice = overrides.get(lang, default_voice)
            logger.info(
                "Loading Kokoro pipeline lang=%s (kokoro_code=%s, voice=%s)",
                lang,
                kokoro_code,
                voice,
            )
            self._pipelines[lang] = KPipeline(lang_code=kokoro_code)
            self._voices[lang] = voice

    @property
    def supported_languages(self) -> set[str]:
        return set(self._pipelines.keys())

    async def synthesize(self, text: str, lang: str) -> bytes:
        if lang not in self._pipelines:
            raise TTSError(f"Kokoro pipeline not loaded for lang={lang!r}")
        start = perf_counter()
        audio = await asyncio.to_thread(self._synthesize_blocking, text, lang)
        logger.debug(
            "Kokoro synthesize: lang=%s text_chars=%d audio_bytes=%d elapsed_ms=%.1f",
            lang,
            len(text),
            len(audio),
            (perf_counter() - start) * 1000,
        )
        return audio

    def _synthesize_blocking(self, text: str, lang: str) -> bytes:
        pipeline = self._pipelines[lang]
        voice = self._voices[lang]

        compute_start = perf_counter()
        chunks: list[np.ndarray] = []
        try:
            for _, _, audio in pipeline(text, voice=voice):
                arr = (
                    audio.detach().cpu().numpy()
                    if hasattr(audio, "detach")
                    else np.asarray(audio)
                )
                chunks.append(arr.astype(np.float32, copy=False))
        except Exception as e:
            raise TTSError(f"Kokoro synthesis failed: {e}") from e

        if not chunks:
            raise TTSError("Kokoro produced no audio chunks")

        waveform = np.concatenate(chunks)
        # Soundfile writes float32 directly; convert to 16-bit PCM WAV.
        buffer = io.BytesIO()
        sf.write(buffer, waveform, _SAMPLE_RATE, format="WAV", subtype="PCM_16")
        audio_bytes = buffer.getvalue()
        # Compute time (on the worker thread) separated from drain/dispatch wait
        # logged in synthesize(); audio_seconds lets us derive the real-time
        # factor (compute_ms vs audio duration) when tuning TTS performance.
        logger.debug(
            "Kokoro compute: lang=%s voice=%s text_chars=%d kokoro_chunks=%d "
            "audio_seconds=%.2f audio_bytes=%d compute_ms=%.1f",
            lang,
            voice,
            len(text),
            len(chunks),
            len(waveform) / _SAMPLE_RATE,
            len(audio_bytes),
            (perf_counter() - compute_start) * 1000,
        )
        return audio_bytes

    async def aclose(self) -> None:
        self._pipelines.clear()
        self._voices.clear()
