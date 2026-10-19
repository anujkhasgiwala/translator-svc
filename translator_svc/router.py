"""POST /v1/translate router.

Thin controller. Parses headers + form, validates inputs before the stream
starts, then hands off to TranslationService. Mid-stream errors are converted
to 0xFF ERROR frames; pre-stream errors propagate to the exception handler.
"""

import asyncio
import logging
from collections.abc import AsyncIterator
from uuid import UUID

from fastapi import Depends  # noqa: E402
from fastapi import APIRouter, File, Form, Header, UploadFile
from fastapi.responses import StreamingResponse
from typing_extensions import Annotated

from translator_svc.config import Settings, get_settings

from .dependencies import get_frame_encoder, get_translation_service
from .errors import (
    AudioTooLargeError,
    InvalidRequestError,
    TranslatorError,
    UnsupportedLanguageError,
)
from .framing import FrameEncoder
from .service import TranslationService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/translate", tags=["translate"])


@router.post(
    "",
    operation_id="translateAudio",
    response_class=StreamingResponse,
    summary="Translate one audio phrase",
)
async def translate_audio(
    audio: Annotated[UploadFile, File(description="Audio bytes for one phrase")],
    source_lang: Annotated[
        str, Form(description="Source language name, e.g. 'English'")
    ],
    target_lang: Annotated[
        str, Form(description="Target language name, e.g. 'Spanish'")
    ],
    x_session_id: Annotated[UUID, Header(alias="X-Session-Id")],
    x_seq: Annotated[int, Header(alias="X-Seq")],
    service: Annotated[TranslationService, Depends(get_translation_service)],
    settings: Annotated[Settings, Depends(get_settings)],
    frames: Annotated[FrameEncoder, Depends(get_frame_encoder)],
) -> StreamingResponse:
    audio_bytes = await audio.read()
    _validate_size(audio_bytes, settings.max_audio_bytes)
    _validate_languages(source_lang, target_lang, settings)
    _validate_seq(x_seq)

    audio_mime = audio.content_type or "audio/wav"
    session_id = str(x_session_id)

    async def stream() -> AsyncIterator[bytes]:
        try:
            async for frame in service.translate_streaming(
                audio_bytes=audio_bytes,
                audio_mime=audio_mime,
                source_lang=source_lang,
                target_lang=target_lang,
                session_id=session_id,
                seq=x_seq,
            ):
                yield frame
        except asyncio.CancelledError:
            logger.info(
                "Client disconnected mid-stream",
                extra={"session_id": session_id, "seq": x_seq},
            )
            raise
        except TranslatorError as e:
            logger.warning(
                "Translator error mid-stream",
                extra={"code": e.code, "session_id": session_id, "seq": x_seq},
            )
            yield frames.encode_error(e.code, e.message)
        except Exception:
            logger.exception(
                "Unexpected pipeline error",
                extra={"session_id": session_id, "seq": x_seq},
            )
            yield frames.encode_error("pipeline_error", "Internal error")

    return StreamingResponse(stream(), media_type="application/octet-stream")


def _validate_size(audio_bytes: bytes, max_bytes: int) -> None:
    if not audio_bytes:
        raise InvalidRequestError("Empty audio body")
    if len(audio_bytes) > max_bytes:
        raise AudioTooLargeError(
            f"Audio is {len(audio_bytes)} bytes; limit is {max_bytes}"
        )


def _validate_languages(source_lang: str, target_lang: str, settings: Settings) -> None:
    if source_lang not in settings.supported_languages:
        raise UnsupportedLanguageError(f"Unsupported source_lang: {source_lang!r}")
    if target_lang not in settings.supported_languages:
        raise UnsupportedLanguageError(f"Unsupported target_lang: {target_lang!r}")
    if source_lang == target_lang:
        raise InvalidRequestError("source_lang and target_lang must differ")


def _validate_seq(seq: int) -> None:
    if seq < 0:
        raise InvalidRequestError(f"X-Seq must be non-negative; got {seq}")
