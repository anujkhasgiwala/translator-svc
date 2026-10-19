#!/usr/bin/env python3
"""Phase 2 smoke test for Kokoro TTS.

Synthesizes a known phrase in each supported language and writes WAV files to
/tmp so you can play them and verify quality.

Run setup first:
    ./scripts/setup_kokoro.sh

Then:
    python scripts/smoke_kokoro.py
    afplay /tmp/translator_test/kokoro_en.wav   # macOS
    afplay /tmp/translator_test/kokoro_es.wav
"""

import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_engine():
    sys.path.insert(0, str(REPO_ROOT))
    from translator_svc.tts.kokoro_local import KokoroLocalTTSEngine

    return KokoroLocalTTSEngine


async def main() -> int:
    out_dir = Path("/tmp/translator_test")
    out_dir.mkdir(exist_ok=True)

    KokoroLocalTTSEngine = _load_engine()

    print(
        "Loading Kokoro pipelines for English, Spanish "
        "(first run downloads weights) ..."
    )
    engine = KokoroLocalTTSEngine(languages=["English", "Spanish"])
    print(f"Supported languages: {sorted(engine.supported_languages)}")

    cases = [
        ("English", "Hello, how are you today?", out_dir / "kokoro_en.wav"),
        ("Spanish", "Hola, ¿cómo estás hoy?", out_dir / "kokoro_es.wav"),
    ]
    for lang, text, out_path in cases:
        print(f"\nSynthesizing [{lang}]: {text!r}")
        audio = await engine.synthesize(text, lang)
        out_path.write_bytes(audio)
        print(f"  wrote {len(audio)} bytes -> {out_path}")

    await engine.aclose()
    print("\nPlay them with:")
    for _, _, p in cases:
        print(f"  afplay {p}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
