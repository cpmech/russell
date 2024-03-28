#!/bin/bash

# the first argument is the "mkl" option
BLAS_LIB=${1:-""}

FEAT=""
if [ "${BLAS_LIB}" = "mkl" ]; then
    FEAT="--features intel_mkl"
fi

cargo build $FEAT

VALGRIND="cargo valgrind run $FEAT"

$VALGRIND --bin mem_check

$VALGRIND --bin solve_matrix_market -- data/matrix_market/bfwb62.mtx -d -g klu
$VALGRIND --bin solve_matrix_market -- data/matrix_market/bfwb62.mtx -d -g mumps
$VALGRIND --bin solve_matrix_market -- data/matrix_market/bfwb62.mtx -d -g umfpack

$VALGRIND --example nonlinear_system_4eqs -- -g klu
$VALGRIND --example nonlinear_system_4eqs -- -g mumps
$VALGRIND --example nonlinear_system_4eqs -- -g umfpack
