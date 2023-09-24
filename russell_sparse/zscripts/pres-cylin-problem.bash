#!/bin/bash

set -e

ODIR=/tmp/pres-cylin
mkdir -p $ODIR

#HASHES="a94e172 379a2db 99506d1"

HASH=`git show --oneline -s | awk '{print $1}'`
MFILE=`find ./src/bin -iname "solve_matrix_market*"`
BIN=solve_matrix_market
OPT1="-g umfpack"
OPT2="-g mumps"

if [ "$MFILE" = "./src/bin/solve_matrix_market_build.rs" ]; then
    BIN=solve_matrix_market_build
    OPT1=""
    OPT2="-m"
fi

cargo clean
cargo build --release

cargo run --release --bin $BIN -- good.mtx $OPT1 > $ODIR/pres-cylin-umfpack-good-local-$HASH.json
cargo run --release --bin $BIN -- bad.mtx  $OPT1 > $ODIR/pres-cylin-umfpack-bad-local-$HASH.json
cargo run --release --bin $BIN -- good.mtx $OPT2 > $ODIR/pres-cylin-mumps-good-local-$HASH.json
cargo run --release --bin $BIN -- bad.mtx  $OPT2 > $ODIR/pres-cylin-mumps-bad-local-$HASH.json

hyperfine "~/rust_modules/release/$BIN good.mtx $OPT1"
hyperfine "~/rust_modules/release/$BIN bad.mtx  $OPT1"
hyperfine "~/rust_modules/release/$BIN good.mtx $OPT2"
hyperfine "~/rust_modules/release/$BIN bad.mtx  $OPT2"
