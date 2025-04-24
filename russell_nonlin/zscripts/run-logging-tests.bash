#!/bin/bash

cargo test --features intel_mkl,local_suitesparse --test test_arc_linear_problem -- --test-threads=1 --nocapture > data/logs/test_arc_linear_problem.txt
cargo test --features intel_mkl,local_suitesparse --test test_arc_single_eq_with_fold -- --test-threads=1 --nocapture > data/logs/test_arc_single_eq_with_fold.txt
cargo test --features intel_mkl,local_suitesparse --test test_linear_problem -- --test-threads=1 --nocapture > data/logs/test_linear_problem.txt
cargo test --features intel_mkl,local_suitesparse --test test_newton_problems -- --test-threads=1 --nocapture > data/logs/test_newton_problems.txt
cargo test --features intel_mkl,local_suitesparse --test test_newton_problems_auto -- --test-threads=1 --nocapture > data/logs/test_newton_problems_auto.txt
