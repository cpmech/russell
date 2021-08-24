#!/bin/bash

set -e

echo "Install:"
echo
echo "cargo install cargo-tarpaulin"
echo "pip3 install pycobertura"
echo

cd zcoverage

cargo +nightly tarpaulin \
    --all \
    --out Html \
    --out Xml

pycobertura show \
    --format=html \
    --output cobertura.html \
    cobertura.xml

browse tarpaulin-report.html
browse cobertura.html

cd ..
