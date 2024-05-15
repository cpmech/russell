#!/bin/bash

INTEL_MKL=${1:-""} # 0 or 1 to use intel_mkl

FEAT=""
if [ "${INTEL_MKL}" = "1" ]; then
    FEAT="--features intel_mkl,local_suitesparse"
fi

for example in examples/*.rs; do
    filename="$(basename "$example")"
    filekey="${filename%%.*}"
    cargo run --release $FEAT --example $filekey
done
