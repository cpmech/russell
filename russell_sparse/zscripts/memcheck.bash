#!/bin/bash

LOCAL_SUITESPARSE=${1:-""}
WITH_MUMPS=${2:-""}
INTEL_MKL=${3:-""}

FEATURES=""
if [ "$LOCAL_SUITESPARSE" = "1" ]; then
    FEATURES="${FEATURES},local_suitesparse"
fi
if [ "$WITH_MUMPS" = "1" ]; then
    FEATURES="${FEATURES},with_mumps"
fi
if [ "$INTEL_MKL" = "1" ]; then
    FEATURES="${FEATURES},intel_mkl"
fi

cargo build --features "$FEATURES"

VALGRIND="cargo valgrind run --features ${FEATURES}"

$VALGRIND --bin mem_check

$VALGRIND --bin solve_matrix_market -- data/matrix_market/bfwb62.mtx -d -g klu
$VALGRIND --bin solve_matrix_market -- data/matrix_market/bfwb62.mtx -d -g umfpack

$VALGRIND --example nonlinear_system_4eqs -- -g klu
$VALGRIND --example nonlinear_system_4eqs -- -g umfpack

if [ "$WITH_MUMPS" = "1" ]; then
    $VALGRIND --bin solve_matrix_market -- data/matrix_market/bfwb62.mtx -d -g mumps
    $VALGRIND --example nonlinear_system_4eqs -- -g mumps
fi
