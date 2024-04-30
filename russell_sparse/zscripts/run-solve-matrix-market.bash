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

cargo run --release --features "$FEATURES" --bin solve_matrix_market -- \
    data/matrix_market/bfwb62.mtx

if [ "$WITH_MUMPS" = "1" ]; then
    cargo run --release --features "$FEATURES" --bin solve_matrix_market -- \
    data/matrix_market/bfwb62.mtx \
    --genie mumps
fi
