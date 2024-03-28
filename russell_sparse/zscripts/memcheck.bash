#!/bin/bash

INTEL_MKL=${1:-""} # 0 or 1 to use intel_mkl

FEAT=""
if [ "${INTEL_MKL}" = "1" ]; then
    FEAT="--features intel_mkl"
fi

cargo build $FEAT

VALGRIND="cargo valgrind run $FEAT"

$VALGRIND --bin mem_check

$VALGRIND --bin solve_matrix_market -- data/matrix_market/bfwb62.mtx -d
$VALGRIND --bin solve_matrix_market -- data/matrix_market/bfwb62.mtx -d -g mumps

$VALGRIND --example nonlinear_system_4eqs --
$VALGRIND --example nonlinear_system_4eqs -- -g mumps
