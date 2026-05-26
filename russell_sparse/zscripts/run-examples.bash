#!/bin/bash

INTEL_MKL=${1:-""}
LOCAL_SPARSE=${2:-""}

FEATURES=""
if [ "$INTEL_MKL" = "1" ]; then
    FEATURES="${FEATURES},intel_mkl"
fi
if [ "$LOCAL_SPARSE" = "1" ]; then
    FEATURES="${FEATURES},local_sparse"
fi

for example in examples/*.rs; do
    filename="$(basename "$example")"
    filekey="${filename%%.*}"
    cargo run --features "$FEATURES" --example $filekey
done
