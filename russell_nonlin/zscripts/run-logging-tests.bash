#!/bin/bash

cargo test --features intel_mkl,local_suitesparse --test test_arc_linear_problem -- --test-threads=1 --nocapture > data/logs/test_arc_linear_problem.txt
cargo test --features intel_mkl,local_suitesparse --test test_arc_one_eq_with_fold -- --test-threads=1 --nocapture > data/logs/test_arc_one_eq_with_fold.txt
cargo test --features intel_mkl,local_suitesparse --test test_arc_singular_initial_state -- --test-threads=1 --nocapture > data/logs/test_arc_singular_initial_state.txt
cargo test --features intel_mkl,local_suitesparse --test test_linear_problem -- --test-threads=1 --nocapture > data/logs/test_linear_problem.txt
cargo test --features intel_mkl,local_suitesparse --test test_newton_problems -- --test-threads=1 --nocapture > data/logs/test_newton_problems.txt
cargo test --features intel_mkl,local_suitesparse --test test_newton_problems_auto -- --test-threads=1 --nocapture > data/logs/test_newton_problems_auto.txt

cargo test --features intel_mkl,local_suitesparse --test test_arc_circle -- --test-threads=1
cp /tmp/russell_nonlin/test_arc_circle_max_lambda.txt data/logs/
cp /tmp/russell_nonlin/test_arc_circle_max_lambda_num_jac.txt data/logs/

cargo test --features intel_mkl,local_suitesparse --test test_circle -- --test-threads=1
cp /tmp/russell_nonlin/test_circle_max_lambda.txt data/logs/
cp /tmp/russell_nonlin/test_circle_max_lambda_num_jac.txt data/logs/
