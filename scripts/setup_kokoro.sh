#!/usr/bin/env bash
# Install Kokoro TTS local deps for the translator service.
#
# Idempotent: safe to re-run.
#
# What this does:
#   1. Installs espeak-ng (system dep) via brew on macOS
#   2. Installs the kokoro + soundfile Python packages into the active env
#      (use `uv pip install` or `pip install`; if neither is in PATH, fails)
#   3. The first `KPipeline(lang_code=...)` call downloads model weights to
#      the HuggingFace cache (~/.cache/huggingface/hub). We don't pre-download
#      here because the package handles caching transparently.

set -euo pipefail

log() { printf '\033[1;34m[setup-kokoro]\033[0m %s\n' "$*"; }
err() { printf '\033[1;31m[setup-kokoro]\033[0m %s\n' "$*" >&2; }

# --- espeak-ng -----------------------------------------------------------

install_espeak() {
    if command -v espeak-ng >/dev/null 2>&1; then
        log "espeak-ng already installed: $(command -v espeak-ng)"
        return
    fi

    if [[ "$(uname)" == "Darwin" ]]; then
        if ! command -v brew >/dev/null 2>&1; then
            err "Homebrew not found. Install from https://brew.sh and re-run."
            exit 1
        fi
        log "Installing espeak-ng via Homebrew..."
        brew install espeak-ng
    else
        err "Automatic install only supported on macOS. On Linux:"
        err "  apt-get install -y espeak-ng    # Debian/Ubuntu"
        err "  dnf install -y espeak-ng        # Fedora"
        exit 1
    fi
}

# --- Python packages -----------------------------------------------------

install_python_deps() {
    # Prefer uv (faster installs). Fall back to pip.
    if command -v uv >/dev/null 2>&1; then
        # uv refuses to install into the system Python without a venv. Create
        # one in .venv if missing, then install into it.
        if [[ ! -d .venv ]]; then
            log "Creating .venv ..."
            uv venv .venv
        else
            log ".venv already exists"
        fi
        log "Installing kokoro + soundfile into .venv via uv ..."
        uv pip install --python .venv/bin/python 'kokoro>=0.9.2' 'soundfile>=0.12.1'
        log ""
        log "Activate the venv with:"
        log "  source .venv/bin/activate"
    elif command -v pip >/dev/null 2>&1; then
        log "Installing kokoro + soundfile via pip (no venv) ..."
        pip install 'kokoro>=0.9.2' 'soundfile>=0.12.1'
    else
        err "Neither uv nor pip found on PATH. Activate a Python env and re-run."
        exit 1
    fi
}

# --- Main ---------------------------------------------------------------

install_espeak
install_python_deps

log "Setup complete."
log ""
log "Smoke test:"
log "  python scripts/smoke_kokoro.py"
