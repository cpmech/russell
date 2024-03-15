#!/bin/bash

# the first argument is 0 or 1 to run with --features intel_mkl
INTEL_MKL=${1:-""}

# the second argument is 0 or 1 to save the results to the data dir (and not compare them)
SAVE_DATA=${2:-""}

TESTS="
test_dopri5_arenstorf_debug \
test_dopri5_arenstorf \
test_dopri5_hairer_wanner_eq1 \
test_dopri5_van_der_pol_debug \
test_dopri8_van_der_pol_debug \
test_dopri8_van_der_pol \
test_radau5_amplifier1t \
test_radau5_hairer_wanner_eq1_debug \
test_radau5_hairer_wanner_eq1 \
test_radau5_robertson_debug \
test_radau5_robertson_small_h \
test_radau5_robertson \
test_radau5_van_der_pol_debug \
test_radau5_van_der_pol \
"

sum=""

for test in $TESTS; do
    temporary="/tmp/${test/test_/russell_}.txt"
    correct="data/logs/${test/test_/russell_}.txt"

    if [ "${SAVE_DATA}" = "1" ]; then
        temporary=$correct
    fi

    # run the test
    if [ "${INTEL_MKL}" = "1" ]; then
        echo "running with:  --features russell_lab/intel_mkl"
        cargo test --test $test --features russell_lab/intel_mkl -- --nocapture --quiet > $temporary
    else
        cargo test --test $test -- --nocapture --quiet > $temporary
    fi

    # delete the last two lines (blank line, the cargo output, and a dot)
    sed -i '$d' $temporary
    sed -i '$d' $temporary
    sed -i '$d' $temporary

    # check the difference
    if [ ! "${SAVE_DATA}" = "1" ]; then
        res=`diff -u $temporary $correct`
        sum="${sum}${res}"
    fi
done

if [ "${SAVE_DATA}" = "1" ]; then
    echo "results saved to the data dir"
else
    echo "this should be empty: [${sum}]"
fi
