"""FastAPI application factory for the real-time translator service."""

import logging
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import Settings
from .errors import TranslatorError
from .lifecycle import on_translator_startup
from .router import router as translator_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.async_exit_stack = AsyncExitStack()
    try:
        await on_translator_startup(app)
    except Exception:
        logger.exception("Service startup failed during translator init")
        await app.state.async_exit_stack.aclose()
        raise
    try:
        yield
    finally:
        await app.state.async_exit_stack.aclose()


def create_app(test_config: Optional[Settings] = None) -> FastAPI:
    config = Settings() if test_config is None else test_config
    logging.basicConfig(level=config.log_level.upper())

    app = FastAPI(
        title="real-time-translator-svc",
        description="Realtime audio translation: speech in, translated text and speech out.",
        docs_url="/apidoc/swagger",
        redoc_url="/apidoc/redoc",
        lifespan=lifespan,
        openapi_tags=[
            {
                "name": "translate",
                "description": "Realtime audio translation endpoints",
            },
        ],
    )
    app.state.config = config

    @app.get("/health", include_in_schema=False)
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(translator_router)

    @app.exception_handler(TranslatorError)
    async def handle_translator_error(request: Request, error: TranslatorError):
        logger.warning(
            "Translator error",
            extra={"code": error.code, "path": request.url.path},
        )
        return JSONResponse(
            content={"code": error.code, "message": error.message},
            status_code=error.http_status,
        )

    # Added last so it runs outermost and answers CORS preflights before routing.
    if config.cors_allow_origins or config.cors_allow_origin_regex:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.cors_allow_origins,
            allow_origin_regex=config.cors_allow_origin_regex or None,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    return app
