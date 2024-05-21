#!/bin/bash

OPENBLAS=${1:-""} # 0 or 1 to use openblas instead of intel_mkl

FEAT="--features intel_mkl,local_suitesparse,with_mumps"
if [ "${OPENBLAS}" = "1" ]; then
    FEAT=""
fi

cargo run --release $FEAT --bin brusselator_pde
