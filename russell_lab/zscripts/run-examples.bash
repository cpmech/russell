#!/bin/bash

for example in examples/*.rs; do
    filename="$(basename "$example")"
    filekey="${filename%%.*}"
    cargo run --example $filekey --features intel_mkl
done
