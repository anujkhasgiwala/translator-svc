import uuid

import pytest
from fastapi.testclient import TestClient

from translator_svc.config import Settings
from translator_svc.main import create_app


@pytest.fixture
def client():
    settings = Settings(llm_backend="fake", tts_backend="fake")
    # Entering the context runs the app lifespan (startup/shutdown).
    with TestClient(create_app(settings)) as test_client:
        yield test_client


def _translate(client, **overrides):
    data = {"source_lang": "English", "target_lang": "Spanish", **overrides}
    return client.post(
        "/v1/translate",
        data=data,
        files={"audio": ("phrase.wav", b"\x00" * 1024, "audio/wav")},
        headers={"X-Session-Id": str(uuid.uuid4()), "X-Seq": "1"},
    )


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_translate_streams_frames(client):
    response = _translate(client)
    assert response.status_code == 200
    assert response.content


def test_unsupported_language_is_rejected_before_streaming(client):
    response = _translate(client, target_lang="Klingon")
    assert 400 <= response.status_code < 500
    assert "code" in response.json()


def test_cors_preflight_allowed_for_localhost(client):
    response = client.options(
        "/v1/translate",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"


def test_cors_preflight_rejected_for_other_origin(client):
    response = client.options(
        "/v1/translate",
        headers={
            "Origin": "https://example.com",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert "access-control-allow-origin" not in response.headers
