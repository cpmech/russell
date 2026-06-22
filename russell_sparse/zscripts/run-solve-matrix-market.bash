#!/bin/bash

GENIE=cudss
MATRIX=~/Downloads/matrix-market/pres-cylin-3d-tet10-fine.mtx

FEATURES="intel_mkl,local_sparse,cudss"

cargo run --release --features "$FEATURES" --bin solve_matrix_market -- \
    "$MATRIX" \
    -v -h 0.8 \
    --positive-definite \
    --genie "$GENIE"
