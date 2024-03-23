#!/bin/bash

INTEL_MKL=${1:-""} # 0 or 1 to use intel_mkl

FEAT=""
if [ "${INTEL_MKL}" = "1" ]; then
    FEAT="--features intel_mkl"
fi

cargo run --release $FEAT --bin brusselator_pde
