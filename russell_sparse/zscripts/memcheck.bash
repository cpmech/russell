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

$VALGRIND --bin solve_matrix_market -- data/matrix_market/bfwb62.mtx -d
$VALGRIND --bin solve_matrix_market -- data/matrix_market/bfwb62.mtx -d -g mumps
if [[ "$WITH_DSS" == "1" ]]; then
    $VALGRIND --bin solve_matrix_market -- data/matrix_market/bfwb62.mtx -d --genie dss
fi

$VALGRIND --example nonlinear_system_4eqs --
$VALGRIND --example nonlinear_system_4eqs -- -g mumps
$VALGRIND --example nonlinear_system_4eqs -- -g dss
