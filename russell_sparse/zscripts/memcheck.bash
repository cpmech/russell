#!/bin/bash

INTEL_MKL=${1:-""}
LOCAL_SPARSE=${2:-""}

FEATURES=""
if [ "$INTEL_MKL" = "1" ]; then
    FEATURES="${FEATURES},intel_mkl"
fi
if [ "$LOCAL_SPARSE" = "1" ]; then
    FEATURES="${FEATURES},local_sparse"
fi

cargo build --features "$FEATURES"

VALGRIND="cargo valgrind run --features ${FEATURES}"

$VALGRIND --bin mem_check

$VALGRIND --bin solve_matrix_market -- data/matrix_market/bfwb62.mtx -d -g klu
$VALGRIND --bin solve_matrix_market -- data/matrix_market/bfwb62.mtx -d -g umfpack

$VALGRIND --example nonlinear_system_4eqs -- -g klu
$VALGRIND --example nonlinear_system_4eqs -- -g umfpack

if [ "$LOCAL_SPARSE" = "1" ]; then
    $VALGRIND --bin solve_matrix_market -- data/matrix_market/bfwb62.mtx -d -g mumps
    $VALGRIND --example nonlinear_system_4eqs -- -g mumps
fi
