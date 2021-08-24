#!/bin/bash

set -e

if [[ $CI != "true" ]]; then
    echo "Install:"
    echo
    echo "cargo install cargo-tarpaulin"
    echo
fi

cd zcoverage

cargo +nightly tarpaulin \
    --all \
    --out Html \
    --out Xml

cd ..
