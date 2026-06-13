#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

PORT="${QUARTO_PORT:-4321}"

exec quarto preview . --render all --port "$PORT" "$@"
