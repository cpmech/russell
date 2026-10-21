#!/bin/bash
# Cargo runner for wasm32-wasip1.
#
# WASI gives a program no filesystem, and no environment, unless the runtime
# grants them. Three grants are needed here:
#
# * the current directory, so the test binaries can reach paths relative to
#   where they run (and so a doctest's temporary build directory is writable);
# * the repository root at its absolute path, because the doctest harness
#   compiles in a temporary directory and every data path is then built from
#   CARGO_MANIFEST_DIR, which lives inside the checkout;
# * CARGO_MANIFEST_DIR itself, since a guest that reads it starts with an empty
#   environment and would otherwise panic on it.
#
# The root is derived from this script's location, so the runner works from any
# checkout rather than only from the one it was written in.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Cargo always sets it; the runner can still be called by hand without it.
manifest_env=()
if [ -n "${CARGO_MANIFEST_DIR:-}" ]; then
  manifest_env=(--env CARGO_MANIFEST_DIR)
fi
exec wasmtime --dir=. --dir="$repo_root" "${manifest_env[@]}" "$@"
