# translator-svc

A FastAPI service for real-time spoken-language translation. You send one phrase of
audio and it streams back the transcription, the translation, and the translation
spoken aloud.

- **Speech to text and translation:** Gemma 4 served by a local
  [llama.cpp](https://github.com/ggml-org/llama.cpp) `llama-server` (OpenAI-compatible API)
- **Text to speech:** [Kokoro](https://github.com/hexgrad/kokoro), run locally
- **Session context:** recent phrases are kept per session (in memory) so
  the model can translate follow-up phrases in context

The matching browser widget lives in [translator-ui](https://github.com/anujkhasgiwala/translator-ui).

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Try it with no model or TTS installed (fake backends):
LLM_BACKEND=fake TTS_BACKEND=fake \
  uvicorn translator_svc.main:create_app --factory --port 8000
```

Then open <http://localhost:8000/apidoc/swagger>.

For real translation, start a `llama-server` with a Gemma 4 model, install the
TTS extra, and point the service at it:

```bash
./scripts/setup_kokoro.sh                  # espeak-ng + kokoro (macOS/Linux)
LLM_SERVER_URL=http://localhost:8080 \
  uvicorn translator_svc.main:create_app --factory --port 8000
./scripts/smoke_llama.sh                   # check llama-server is reachable
```

## API

`POST /v1/translate` (multipart form)

| Field / header | Description |
|----------------|-------------|
| `audio` | Audio bytes for one phrase |
| `source_lang`, `target_lang` | Language names, e.g. `English`, `Spanish` |
| `X-Session-Id` | UUID identifying the conversation |
| `X-Seq` | Phrase sequence number within the session |

The response is a binary stream of frames:
`[1 byte type][4 byte big-endian length][payload]`, where type is
`0x01` source text, `0x02` target text, `0x03` audio, or `0xFF` error (JSON).
Errors found before streaming starts return a normal JSON `{code, message}` response.

`GET /health` returns `{"status": "ok"}`.

## Configuration

All settings are environment variables; see `translator_svc/config.py` for the
full list and defaults. The common ones:

| Variable | Default | Purpose |
|----------|---------|---------|
| `LLM_BACKEND` | `llama_cpp` | `llama_cpp` or `fake` |
| `LLM_SERVER_URL` | `http://localhost:8080` | llama-server base URL |
| `LLM_MODEL_VARIANT` | `e4b` | Gemma 4 variant: `e4b` or `e2b` |
| `TTS_BACKEND` | `kokoro_local` | `kokoro_local` or `fake` |
| `SESSION_STORE_BACKEND` | `memory` | `memory` (the `redis` backend is a stub, not implemented yet) |
| `SUPPORTED_LANGUAGES` | English, Spanish, Hindi, French, Italian | Languages accepted by the API |
| `CORS_ALLOW_ORIGINS` | none | Comma-separated browser origins |
| `CORS_ALLOW_ORIGIN_REGEX` | localhost on any port | Regex alternative to the list above |

## Development

```bash
pytest          # unit tests plus app-level tests using the fake backends
mypy translator_svc tests
black . && isort .
```

## Docker

```bash
docker build -t translator-svc .
docker run -p 8000:8000 -e LLM_BACKEND=fake -e TTS_BACKEND=fake translator-svc
```

## License

MIT, see [LICENSE](LICENSE).
