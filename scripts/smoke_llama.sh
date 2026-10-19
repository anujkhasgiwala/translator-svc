#!/usr/bin/env bash
# Smoke-test a running llama-server: probe /health and /v1/models.
#
# Environment:
#   LLAMA_URL  - base URL (default: http://127.0.0.1:8080)

set -euo pipefail

LLAMA_URL="${LLAMA_URL:-http://127.0.0.1:8080}"

log() { printf '\033[1;34m[smoke-llama]\033[0m %s\n' "$*"; }
err() { printf '\033[1;31m[smoke-llama]\033[0m %s\n' "$*" >&2; }

log "Probing $LLAMA_URL/health ..."
if ! curl -sS --fail --max-time 5 "$LLAMA_URL/health" -o /tmp/llama_health.json; then
    err "Health probe failed. Is llama-server running on $LLAMA_URL?"
    exit 1
fi
log "  $(cat /tmp/llama_health.json)"

log "Probing $LLAMA_URL/v1/models ..."
if ! curl -sS --fail --max-time 5 "$LLAMA_URL/v1/models" -o /tmp/llama_models.json; then
    err "Models probe failed."
    exit 1
fi
log "  $(cat /tmp/llama_models.json)"

log "llama-server is alive."
