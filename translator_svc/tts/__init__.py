"""TTSEngine factory."""

from translator_svc.config import Settings

from .base import TTSEngine
from .fake import FakeTTSEngine

__all__ = ["TTSEngine", "make_tts_engine"]


def make_tts_engine(settings: Settings) -> TTSEngine:
    backend = settings.tts_backend
    if backend == "kokoro_local":
        # Imported lazily: kokoro/torch are an optional extra (`pip install .[kokoro]`).
        from .kokoro_local import KokoroLocalTTSEngine

        return KokoroLocalTTSEngine(
            model_path=settings.tts_model_path,
            languages=settings.supported_languages,
        )
    if backend == "fake":
        return FakeTTSEngine()
    raise ValueError(f"Unsupported tts_backend: {backend!r}")
