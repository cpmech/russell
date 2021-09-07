#!/bin/bash

cargo valgrind run --bin mem_check_build
cargo valgrind run --bin solve_mm_build -- data/matrix_market/bfwb62.mtx
cargo valgrind run --bin solve_mm_build -- data/matrix_market/bfwb62.mtx --mmp
