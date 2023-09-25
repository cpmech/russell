#!/bin/bash

if [[ -z "${RUSSELL_SPARSE_WITH_INTEL_DSS}" ]]; then
  WITH_DSS="0"
else
  WITH_DSS="${RUSSELL_SPARSE_WITH_INTEL_DSS}"
fi

cargo build

cargo valgrind run --bin mem_check
cargo valgrind run --bin solve_matrix_market -- data/matrix_market/bfwb62.mtx
cargo valgrind run --bin solve_matrix_market -- data/matrix_market/bfwb62.mtx --genie mumps -n 1

if [[ "$WITH_DSS" == "1" ]]; then
    cargo valgrind run --bin solve_matrix_market -- data/matrix_market/bfwb62.mtx --genie dss
fi
