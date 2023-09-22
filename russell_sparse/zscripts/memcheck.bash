#!/bin/bash

cargo valgrind run --bin mem_check
cargo valgrind run --bin solve_matrix_market -- data/matrix_market/bfwb62.mtx
cargo valgrind run --bin solve_matrix_market -- data/matrix_market/bfwb62.mtx --genie mumps -n 1
