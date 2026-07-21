#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)
BUILD_DIR="$SCRIPT_DIR/.build"
TTS_VENV="$BUILD_DIR/tts-venv"
SCREENCAST_DIR=${OA_SCREENCAST_DIR:-"$REPO_ROOT/var/build-week/screencasts"}

for command in ffmpeg ffprobe magick node python3 curl; do
    command -v "$command" >/dev/null || {
        printf 'Missing required command: %s\n' "$command" >&2
        exit 1
    }
done

mkdir -p "$BUILD_DIR"
if [[ ! -x "$TTS_VENV/bin/edge-tts" ]]; then
    python3 -m venv "$TTS_VENV"
    "$TTS_VENV/bin/python" -m pip install --quiet edge-tts
fi

node "$SCRIPT_DIR/render.mjs" \
    --repo "$REPO_ROOT" \
    --build "$BUILD_DIR" \
    --screencasts "$SCREENCAST_DIR" \
    --edge-tts "$TTS_VENV/bin/edge-tts"
