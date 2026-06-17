#!/bin/bash

GENIE=${1:-"umfpack"}

FEATURES="intel_mkl,local_sparse,cudss"

cargo run --release --features "$FEATURES" --bin solve_matrix_market -- \
    data/matrix_market/bfwb62.mtx \
    --genie $GENIE
