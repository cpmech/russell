#!/bin/bash

cargo build --release

cargo run --release --bin solve_matrix_market -- data/matrix_market/bfwb62.mtx
cargo run --release --bin solve_matrix_market -- data/matrix_market/bfwb62.mtx --genie mumps
cargo run --release --bin solve_matrix_market -- data/matrix_market/bfwb62.mtx --genie dss

#cargo run --release --bin solve_matrix_market -- data/matrix_market/bfwb62.mtx > data/results/solve-matrix-market-umfpack-bfwb62.json
#cargo run --release --bin solve_matrix_market -- data/matrix_market/bfwb62.mtx --genie mumps > data/results/solve-matrix-market-mumps-bfwb62.json
#cargo run --release --bin solve_matrix_market -- data/matrix_market/bfwb62.mtx --genie dss > data/results/solve-matrix-market-dss-bfwb62.json
