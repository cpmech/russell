#!/bin/bash

LOCAL_SUITESPARSE=${1:-""}
WITH_MUMPS=${2:-""}
INTEL_MKL=${3:-""}

FEATURES=""
if [ "$LOCAL_SUITESPARSE" = "1" ]; then
    FEATURES="${FEATURES},local_suitesparse"
fi
if [ "$WITH_MUMPS" = "1" ]; then
    FEATURES="${FEATURES},with_mumps"
fi
if [ "$INTEL_MKL" = "1" ]; then
    FEATURES="${FEATURES},intel_mkl"
fi

for example in examples/*.rs; do
    filename="$(basename "$example")"
    filekey="${filename%%.*}"
    cargo run --features "$FEATURES" --example $filekey
done
