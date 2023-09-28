#!/bin/bash

if [[ -z "${RUSSELL_SPARSE_WITH_INTEL_DSS}" ]]; then
  WITH_DSS="0"
else
  WITH_DSS="${RUSSELL_SPARSE_WITH_INTEL_DSS}"
fi

cargo build --release

cargo run --release --bin solve_matrix_market -- data/matrix_market/bfwb62.mtx
cargo run --release --bin solve_matrix_market -- data/matrix_market/bfwb62.mtx --genie mumps

if [[ "$WITH_DSS" == "1" ]]; then
    cargo run --release --bin solve_matrix_market -- data/matrix_market/bfwb62.mtx --genie dss
fi

#cargo run --release --bin solve_matrix_market -- data/matrix_market/bfwb62.mtx > data/results/solve-matrix-market-umfpack-bfwb62.json
#cargo run --release --bin solve_matrix_market -- data/matrix_market/bfwb62.mtx --genie mumps > data/results/solve-matrix-market-mumps-bfwb62.json

#if [[ "$WITH_DSS" == "1" ]]; then
    #cargo run --release --bin solve_matrix_market -- data/matrix_market/bfwb62.mtx --genie dss > data/results/solve-matrix-market-dss-bfwb62.json
#fi
