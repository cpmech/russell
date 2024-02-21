#!/bin/bash

TESTS="
test_radau5_hairer_wanner_eq1_dense \
test_radau5_hairer_wanner_eq1 \
test_radau5_van_der_pol_dense \
test_radau5_van_der_pol \
"

sum=""

for test in $TESTS; do
    correct="data/${test/test_/diff_}.txt"
    fortran="data/${test/test_/fortran_}.txt"
    russell="data/${test/test_/russell_}.txt"

    # run the test
    cargo test --test $test -- --nocapture --quiet > $russell 

    # delete the last two lines (blank line, the cargo output, and a dot)
    sed -i '$d' $russell
    sed -i '$d' $russell
    sed -i '$d' $russell

    # temporary diff (run this just once)
    # pipe to tail to ignore the first two lines because of the date/time
    # diff -u $fortran $russell | tail -n +3 > $correct

    res=`diff -u $fortran $russell | tail -n +3 | diff - $correct`
    sum="${sum}${res}"
done

echo "this should be empty: [${sum}]"
