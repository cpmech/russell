#!/bin/bash

TESTS="
test_dopri5_arenstorf_debug \
test_dopri5_arenstorf \
"

sum=""

for test in $TESTS; do
    temporary="/tmp/${test/test_/russell_}.txt"
    correct="data/${test/test_/russell_}.txt"

    # run the test
    cargo test --test $test -- --nocapture --quiet > $temporary

    # delete the last two lines (blank line, the cargo output, and a dot)
    sed -i '$d' $temporary
    sed -i '$d' $temporary
    sed -i '$d' $temporary

    # check the difference
    res=`diff -u $temporary $correct`
    sum="${sum}${res}"
done

echo "this should be empty: [${sum}]"
