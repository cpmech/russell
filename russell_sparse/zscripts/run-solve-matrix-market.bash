#!/bin/bash

# the first argument is the "mkl" option
BLAS_LIB=${1:-""}

FEAT=""
if [ "${BLAS_LIB}" = "mkl" ]; then
    FEAT="--features intel_mkl"
fi

cargo build --release $FEAT

cargo run --release $FEAT --bin solve_matrix_market -- data/matrix_market/bfwb62.mtx
cargo run --release $FEAT --bin solve_matrix_market -- data/matrix_market/bfwb62.mtx --genie mumps
