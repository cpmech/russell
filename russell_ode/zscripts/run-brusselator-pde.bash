#!/bin/bash

SECOND_BOOK=${1:-""} # 0 or 1 for the second book
ONLY_PLOT=${2:-""} # 0 or 1 for only plot (using existent results)
INTEL_MKL=${3:-""} # 0 or 1 to use intel_mkl

FEAT=""
if [ "${INTEL_MKL}" = "1" ]; then
    FEAT="--features intel_mkl"
fi

if [ ! "${ONLY_PLOT}" = "1" ]; then
    if [ "${SECOND_BOOK}" = "1" ]; then
        echo "BOOK2 CALCULATIONS"
        cargo run --release $FEAT --example brusselator_pde_radau5_2nd
    else
        echo "BOOK1 CALCULATIONS"
        cargo run --release $FEAT --example brusselator_pde_radau5
    fi
fi

if [ "${SECOND_BOOK}" = "1" ]; then
    echo "BOOK2 PLOT"
    cargo run --release $FEAT --example brusselator_pde_plot -- --second-book
else
    echo "BOOK1 PLOT"
    cargo run --release $FEAT --example brusselator_pde_plot
fi
