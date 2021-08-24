#!/bin/bash

set -e

echo "Install:"
echo
echo "cargo install cargo-tarpaulin"
echo "pip3 install pycobertura"
echo

OUTDIR=/tmp/russell

cd zcoverage

cargo tarpaulin \
    --all \
    --out Html \
    --out Xml \
    --target-dir=$OUTDIR \
    --output-dir=$OUTDIR

pycobertura show \
    --format=html \
    --output $OUTDIR/cobertura.html \
    $OUTDIR/cobertura.xml

browse $OUTDIR/tarpaulin-report.html
browse $OUTDIR/cobertura.html

cd ..
