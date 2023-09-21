#!/bin/bash

cargo run --release --bin solve_matrix_market_build -- data/matrix_market/bfwb62.mtx
cargo run --release --bin solve_matrix_market_build -- data/matrix_market/bfwb62.mtx --mumps
