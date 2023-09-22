#!/bin/bash

cargo run --release --bin solve_matrix_market -- data/matrix_market/bfwb62.mtx
cargo run --release --bin solve_matrix_market -- data/matrix_market/bfwb62.mtx --genie mumps
