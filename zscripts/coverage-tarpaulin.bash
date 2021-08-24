#!/bin/bash

set -e

if [[ $CI != "true" ]]; then
    echo "Install:"
    echo
    echo "cargo install cargo-tarpaulin"
    echo
fi

cd zcoverage

if [[ $CI != "true" ]]; then
    cargo +nightly tarpaulin \
        --all \
        --out Html \
        --out Xml \
        --fail-under 95
else
    cargo +nightly tarpaulin \
        --all \
        --out Html \
        --out Xml
fi

cd ..
