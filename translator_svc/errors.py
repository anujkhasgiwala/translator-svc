"""Translator-specific exception hierarchy.

Each exception carries a stable `code` for wire-level error frames and an
`http_status` for pre-stream JSON error responses. The router decides which
representation to use based on whether the response has started streaming.
"""


class TranslatorError(Exception):
    code: str = "translator_error"
    http_status: int = 500

    def __init__(self, message: str = "") -> None:
        super().__init__(message or self.code)
        self.message = message or self.code


class InvalidRequestError(TranslatorError):
    code = "invalid_request"
    http_status = 400


class UnsupportedLanguageError(InvalidRequestError):
    code = "unsupported_language"
    http_status = 422


class AudioTooLargeError(InvalidRequestError):
    code = "audio_too_large"
    http_status = 413


class LLMUnavailableError(TranslatorError):
    code = "llm_unavailable"
    http_status = 503


class LLMOutputParseError(TranslatorError):
    code = "llm_output_parse_error"
    http_status = 502


class TTSError(TranslatorError):
    code = "tts_error"
    http_status = 500


class PipelineError(TranslatorError):
    code = "pipeline_error"
    http_status = 500
