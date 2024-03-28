#!/bin/bash

INTEL_MKL=${1:-""} # 0 or 1 to use intel_mkl
LOCAL_LIBS=${2:-""} # 0 or 1 to use local libs

FEAT=""
if [ "${INTEL_MKL}" = "1" ]; then
    FEAT="--features intel_mkl"
fi
if [ "${LOCAL_LIBS}" = "1" ]; then
    FEAT="${FEAT} --features local_libs"
fi

cargo build --release $FEAT

cargo run --release $FEAT --bin solve_matrix_market -- data/matrix_market/bfwb62.mtx
cargo run --release $FEAT --bin solve_matrix_market -- data/matrix_market/bfwb62.mtx --genie mumps
