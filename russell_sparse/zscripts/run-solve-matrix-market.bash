#!/bin/bash

INTEL_MKL=${1:-""}
WITH_SPARSE=${2:-""}

FEATURES=""
if [ "$INTEL_MKL" = "1" ]; then
    FEATURES="${FEATURES},intel_mkl"
fi
if [ "$WITH_SPARSE" = "1" ]; then
    FEATURES="${FEATURES},with_sparse"
fi

cargo run --release --features "$FEATURES" --bin solve_matrix_market -- \
    data/matrix_market/bfwb62.mtx

if [ "$WITH_SPARSE" = "1" ]; then
    cargo run --release --features "$FEATURES" --bin solve_matrix_market -- \
    data/matrix_market/bfwb62.mtx \
    --genie mumps
fi
