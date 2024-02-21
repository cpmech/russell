#!/bin/bash

TESTS="
test_radau5_hairer_wanner_eq1_dense \
test_radau5_hairer_wanner_eq1 \
test_radau5_van_der_pol \
"

for test in $TESTS; do
    cargo test --test $test -- --nocapture --quiet > "data/log_$test.txt"
done
