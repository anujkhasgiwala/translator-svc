import pytest

from translator_svc.config import Settings


@pytest.fixture(autouse=True)
def anyio_backend():
    return "asyncio"


@pytest.fixture
def settings():
    return Settings()
