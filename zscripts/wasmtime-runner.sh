#!/bin/bash
# Cargo runner for wasm32-wasip1: preopen the current directory so the
# file-based tests can read the repository data.
set -euo pipefail
exec wasmtime --dir=. "$@"
