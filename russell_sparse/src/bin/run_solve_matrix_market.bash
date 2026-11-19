#!/bin/bash

set -euo pipefail

# Matrices in ascending size, so an interrupted suite still has the small rows
REAL_MATS=(
    pmsim-hex8-27
    bwm2000
    rdb5000
    Goodwin_040
    fp
    xenon1
    twotone
    Raj1
    boyd2
    pres-cylin-3d-tet10-coarse
    Goodwin_071
    darcy003
    rma10
    helm2d03
    stomach
    oilpan
    ASIC_680k
    tmt_unsym
    Goodwin_127
    pre2
    marine1
    torso1
    atmosmodd
    atmosmodl
    memchip
    Freescale1
    rajat31
    Transport
    inline_1
    PFlow_742
    Emilia_923
    dielFilterV2real
    Flan_1565
    pres-cylin-3d-tet10-fine
)

COMPLEX_MATS=(
    mhd1280b
    mplate
    RFdevice
    vfem
    fem_filter
    Chevron4
    mono_500Hz
    kim2
    fem_hifreq_circuit
    dielFilterV3clx
)

HERE=$(cd "$(dirname "$0")" && pwd)
RUSSELL_DIR=${RUSSELL_DIR:-$(cd "$HERE/../../.." && pwd)}
INPUT_DIR=${INPUT_DIR:-$HOME/Downloads/matrix-market}
SYSTEM=${SYSTEM:-arch}
OUTPUT_DIR=${OUTPUT_DIR:-/tmp/russell_sparse/results/$SYSTEM}
FEATURES=${FEATURES-intel_mkl,umfpack,mumps}
GENIE=${GENIE:-FAER}

usage() {
    echo "Usage: $0 [--information | --solve] [--first10]"
    echo ""
    echo "  -i, --information  only print the matrix information (default)"
    echo "  -s, --solve        solve the linear systems"
    echo "  -1, --first10      only run the first 10 REAL_MATS"
    echo ""
    echo "Environment: RUSSELL_DIR, INPUT_DIR, OUTPUT_DIR, SYSTEM, FEATURES, GENIE"
}

INFORMATION=1
FIRST10=0
FAILED=0
while (($# > 0)); do
    case "$1" in
        -i | --information) INFORMATION=1 ;;
        -s | --solve) INFORMATION=0 ;;
        -1 | --first10) FIRST10=1 ;;
        -h | --help)
            usage
            exit 0
            ;;
        *)
            echo "unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
    shift
done

if [[ ! -d "$INPUT_DIR" ]]; then
    echo "input directory not found: $INPUT_DIR" >&2
    exit 1
fi
if [[ ! -f "$RUSSELL_DIR/Cargo.toml" ]]; then
    echo "Russell workspace not found: $RUSSELL_DIR" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
INPUT_DIR=$(cd "$INPUT_DIR" && pwd)
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)

RUN_MATS=("${REAL_MATS[@]}" "${COMPLEX_MATS[@]}")
if ((FIRST10)); then
    RUN_MATS=("${REAL_MATS[@]:0:10}")
fi
TOTAL_MATS=${#RUN_MATS[@]}
CURRENT_MAT=0

ARGS=(-g "$GENIE")
if ((INFORMATION)); then
    ARGS+=(--information)
fi

FEATURE_ARGS=()
if [[ -n "$FEATURES" ]]; then
    FEATURE_ARGS=(--features "$FEATURES")
fi

echo "=== Building solve_matrix_market (features: ${FEATURES:-none}) ===" >&2
cargo build --release --manifest-path "$RUSSELL_DIR/Cargo.toml" \
    --package russell_sparse "${FEATURE_ARGS[@]}" --bin solve_matrix_market

run_matrix() {
    local name=$1
    CURRENT_MAT=$((CURRENT_MAT + 1))
    local file="$INPUT_DIR/$name.mtx"
    local out="$OUTPUT_DIR/$name.json"
    if [[ ! -f "$file" ]]; then
        echo "skip $CURRENT_MAT/$TOTAL_MATS $name: $file not found" >&2
        return
    fi
    if ((INFORMATION)); then
        echo "=== $CURRENT_MAT/$TOTAL_MATS $name (information only) ===" >&2
    else
        echo "=== $CURRENT_MAT/$TOTAL_MATS $name ===" >&2
    fi
    if ! cargo run -q --release --manifest-path "$RUSSELL_DIR/Cargo.toml" \
        --package russell_sparse "${FEATURE_ARGS[@]}" \
        --bin solve_matrix_market -- "${ARGS[@]}" "$file" >"$out"; then
        echo "failed: $name" >&2
        FAILED=1
    fi
}

TIMEFORMAT='total time: %3R s'
time {
    for name in "${RUN_MATS[@]}"; do
        run_matrix "$name"
    done
}

if ((FAILED)); then
    echo "some matrices failed; results written to $OUTPUT_DIR" >&2
    exit 1
fi

echo "results written to $OUTPUT_DIR" >&2
