#!/bin/bash

if [[ -z "${RUSSELL_SPARSE_WITH_INTEL_DSS}" ]]; then
  WITH_DSS="0"
else
  WITH_DSS="${RUSSELL_SPARSE_WITH_INTEL_DSS}"
fi

cargo build

cargo valgrind run --bin mem_check

cargo valgrind run --bin solve_matrix_market -- data/matrix_market/bfwb62.mtx -d
cargo valgrind run --bin solve_matrix_market -- data/matrix_market/bfwb62.mtx -d -g mumps
if [[ "$WITH_DSS" == "1" ]]; then
    cargo valgrind run --bin solve_matrix_market -- data/matrix_market/bfwb62.mtx -d --genie dss
fi

cargo valgrind run --example nonlinear_system_4eqs --
cargo valgrind run --example nonlinear_system_4eqs -- -g mumps
if [[ "$WITH_DSS" == "1" ]]; then
    cargo valgrind run --example nonlinear_system_4eqs -- -g dss
fi
