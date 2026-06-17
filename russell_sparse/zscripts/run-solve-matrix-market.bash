#!/bin/bash

GENIE=${1:-"umfpack"}
MATRIX=${2:-"data/matrix_market/bfwb62.mtx"}

FEATURES="intel_mkl,local_sparse,cudss"

cargo run --release --features "$FEATURES" --bin solve_matrix_market -- \
    "$MATRIX" \
    --genie "$GENIE"
