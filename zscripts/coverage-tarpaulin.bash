#!/bin/bash

set -e

if [[ $CI != "true" ]]; then
    echo "Install:"
    echo
    echo "cargo install cargo-tarpaulin"
    echo "pip3 install pycobertura"
    echo
fi

cd zcoverage

cargo +nightly tarpaulin \
    --all \
    --out Html \
    --out Xml

if [[ $CI != "true" ]]; then
    pycobertura show \
        --format=html \
        --output cobertura.html \
        cobertura.xml

    browse tarpaulin-report.html
    browse cobertura.html
fi

cd ..
