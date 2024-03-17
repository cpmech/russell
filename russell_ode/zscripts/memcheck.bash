#!/bin/bash

# the first argument is the "mkl" option
BLAS_LIB=${1:-""}

FEAT=""
if [ "${BLAS_LIB}" = "mkl" ]; then
    FEAT="--features intel_mkl"
fi

cargo build $FEAT

VALGRIND="cargo valgrind run $FEAT"

$VALGRIND --bin amplifier1t_radau5
