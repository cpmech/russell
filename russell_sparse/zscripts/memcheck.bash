#!/bin/bash

# Needs:
# cargo install cargo-valgrind

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SUPPRESSIONS="$PROJECT_DIR/zscripts/valgrind-mumps.supp"

FEATURES="intel_mkl,local_sparse,cudss"
BINARY="$HOME/rust_modules/debug/mem_check"

cd "$PROJECT_DIR"

# Build
echo "=== Building with features: $FEATURES ==="
cargo build --features "$FEATURES" --bin mem_check

# Run valgrind with MUMPS suppressions
echo ""
echo "=== Running valgrind ==="
valgrind \
    --leak-check=full \
    --suppressions="$SUPPRESSIONS" \
    "$BINARY"
