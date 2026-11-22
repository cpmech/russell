#!/bin/bash

set -euo pipefail

# Returns true when the matrix is marked as symmetric positive definite
is_spd() {
    local name=$1
    local candidate
    for candidate in "${SPD_MATS[@]}"; do
        if [[ "$candidate" == "$name" ]]; then
            return 0
        fi
    done
    return 1
}

HERE=$(cd "$(dirname "$0")" && pwd)
RUSSELL_DIR=${RUSSELL_DIR:-$(cd "$HERE/../../.." && pwd)}
INPUT_DIR=${INPUT_DIR:-$HOME/Downloads/matrix-market}
SYSTEM=${SYSTEM:-arch}
OUTPUT_DIR=${OUTPUT_DIR:-/tmp/russell_sparse/results/$SYSTEM}
FEATURES=${FEATURES-intel_mkl,umfpack,mumps}
MATRICES_JSON=${MATRICES_JSON:-"$RUSSELL_DIR/russell_sparse/data/matrix_market/matrices.json"}
SOLVER=FAER

usage() {
    echo "Usage: $0 [--information | --solve] [-g <name>] [-f <n>] [-F <n>] [-1..-5]"
    echo ""
    echo "  -i, --information  only print the matrix information (default)"
    echo "  -s, --solve        solve the linear systems"
    echo "  -g, --genie <name> solver (default FAER)"
    echo "  -G, --all-genies   run all available solvers"
    echo "  -f, --first <n>    only run the first <n> REAL_MATS"
    echo "  -F, --first-complex <n>  only run the first <n> COMPLEX_MATS"
    echo "  -1 ... -5          number of runs (default 1)"
    echo ""
    echo "Environment: RUSSELL_DIR, INPUT_DIR, OUTPUT_DIR, INFO_FILE, MATRICES_JSON, SYSTEM, FEATURES, TIME_GUARD"
}

INFORMATION=1
ALL_GENIES=0
FIRST_N=0
FIRST_COMPLEX_N=0
NRUN=1
TIME_GUARD=${TIME_GUARD:-200}
while (($# > 0)); do
    case "$1" in
        -i | --information) INFORMATION=1 ;;
        -s | --solve) INFORMATION=0 ;;
        -g | --genie)
            if (($# < 2)); then
                echo "option $1 requires a solver name" >&2
                exit 1
            fi
            SOLVER=$2
            shift
            ;;
        -G | --all-genies) ALL_GENIES=1 ;;
        -f | --first)
            if (($# < 2)); then
                echo "option $1 requires a number" >&2
                exit 1
            fi
            if [[ ! $2 =~ ^[1-9][0-9]*$ ]]; then
                echo "option $1 requires a positive integer" >&2
                exit 1
            fi
            FIRST_N=$2
            shift
            ;;
        -F | --first-complex)
            if (($# < 2)); then
                echo "option $1 requires a number" >&2
                exit 1
            fi
            if [[ ! $2 =~ ^[1-9][0-9]*$ ]]; then
                echo "option $1 requires a positive integer" >&2
                exit 1
            fi
            FIRST_COMPLEX_N=$2
            shift
            ;;
        -1) NRUN=1 ;;
        -2) NRUN=2 ;;
        -3) NRUN=3 ;;
        -4) NRUN=4 ;;
        -5) NRUN=5 ;;
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

if [[ ! "$TIME_GUARD" =~ ^[0-9]+$ ]]; then
    echo "invalid TIME_GUARD: $TIME_GUARD (expected a non-negative integer)" >&2
    exit 1
fi

# Build the benchmark matrix lists from the canonical metadata.
if ! command -v jq >/dev/null 2>&1; then
    echo "jq is required to read $MATRICES_JSON" >&2
    exit 1
fi
if [[ ! -f "$MATRICES_JSON" ]]; then
    echo "matrix metadata not found: $MATRICES_JSON" >&2
    exit 1
fi

REAL_MATS=()
while IFS= read -r name; do
    [[ -n "$name" ]] && REAL_MATS+=("$name")
done < <(jq -r '.[] | select(.Type == "real") | .Name' "$MATRICES_JSON")

COMPLEX_MATS=()
while IFS= read -r name; do
    [[ -n "$name" ]] && COMPLEX_MATS+=("$name")
done < <(jq -r '.[] | select(.Type == "complex") | .Name' "$MATRICES_JSON")

# Real symmetric positive-definite matrices get the solver option -p.
SPD_MATS=()
while IFS= read -r name; do
    [[ -n "$name" ]] && SPD_MATS+=("$name")
done < <(jq -r '.[] | select(.Type == "real" and .Symmetric == "Yes" and ."Positive Definite" == "yes") | .Name' "$MATRICES_JSON")

if ((${#REAL_MATS[@]} == 0 || ${#COMPLEX_MATS[@]} == 0)); then
    echo "matrix metadata has no real or complex matrices: $MATRICES_JSON" >&2
    exit 1
fi

if ((ALL_GENIES)); then
    GENIES=()
    if [[ "$FEATURES" == *mumps* ]]; then
        GENIES+=(mumps)
    fi
    if [[ "$FEATURES" == *umfpack* ]]; then
        GENIES+=(umfpack)
    fi
    GENIES+=(faer)
else
    shopt -s nocasematch
    if [[ ! "$SOLVER" =~ ^(faer|mumps|umfpack)$ ]]; then
        echo "invalid solver: $SOLVER (choose one of: faer, mumps, umfpack)" >&2
        exit 1
    fi
    shopt -u nocasematch
    GENIES=("$SOLVER")
fi

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
INFO_FILE=${INFO_FILE:-"$OUTPUT_DIR/system-information.txt"}

REAL_RUN_MATS=()
COMPLEX_RUN_MATS=()
if ((FIRST_N > 0 || FIRST_COMPLEX_N > 0)); then
    if ((FIRST_N > 0)); then
        REAL_RUN_MATS+=("${REAL_MATS[@]:0:FIRST_N}")
    fi
    if ((FIRST_COMPLEX_N > 0)); then
        COMPLEX_RUN_MATS+=("${COMPLEX_MATS[@]:0:FIRST_COMPLEX_N}")
    fi
else
    REAL_RUN_MATS=("${REAL_MATS[@]}")
    COMPLEX_RUN_MATS=("${COMPLEX_MATS[@]}")
fi

FEATURE_ARGS=()
if [[ -n "$FEATURES" ]]; then
    FEATURE_ARGS=(--features "$FEATURES")
fi

RUSSELL_VERSION=$(sed -n 's/^version = "\([^"]*\)"/\1/p' "$RUSSELL_DIR/russell_sparse/Cargo.toml" | head -n 1)
RUSSELL_VERSION=${RUSSELL_VERSION:-unknown}
RUSSELL_COMMIT=$(git -C "$RUSSELL_DIR" rev-parse --short HEAD 2>/dev/null || true)
RUSSELL_COMMIT=${RUSSELL_COMMIT:-unknown}
if [[ "$RUSSELL_COMMIT" != "unknown" ]] && [[ -n "$(git -C "$RUSSELL_DIR" status --porcelain --untracked-files=no 2>/dev/null)" ]]; then
    RUSSELL_COMMIT="${RUSSELL_COMMIT}-dirty"
fi

SELECTED_SPD=()
for name in "${REAL_RUN_MATS[@]}" "${COMPLEX_RUN_MATS[@]}"; do
    if is_spd "$name"; then
        SELECTED_SPD+=("$name")
    fi
done

echo "=== Building solve_matrix_market and sys_info (features: ${FEATURES:-none}) ===" >&2
cargo build --release --manifest-path "$RUSSELL_DIR/Cargo.toml" \
    --package russell_sparse "${FEATURE_ARGS[@]}" --bin solve_matrix_market --bin sys_info

{
    echo "Benchmark configuration"
    echo "date: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    echo "russell version: $RUSSELL_VERSION"
    echo "russell commit: $RUSSELL_COMMIT"
    echo "system: $SYSTEM"
    echo "features: ${FEATURES:-none}"
    echo "solvers: ${GENIES[*]}"
    echo "nrun: $NRUN"
    echo "time_guard: $TIME_GUARD"
    echo "information_only: $INFORMATION"
    echo "input: $INPUT_DIR"
    echo "output: $OUTPUT_DIR"
    echo
    echo "solver arguments:"
    for solver in "${GENIES[@]}"; do
        printf '  -g %s --nrun %s --time-guard %s' "$solver" "$NRUN" "$TIME_GUARD"
        if ((INFORMATION)); then
            printf ' --information'
        fi
        printf '\n'
    done
    echo
    echo "matrix arguments: --positive-definite for the SPD matrices listed below"
    echo
    echo "real matrices (${#REAL_RUN_MATS[@]}):"
    if ((${#REAL_RUN_MATS[@]} > 0)); then
        printf '  %s\n' "${REAL_RUN_MATS[@]}"
    else
        echo "  (none)"
    fi
    echo "complex matrices (${#COMPLEX_RUN_MATS[@]}):"
    if ((${#COMPLEX_RUN_MATS[@]} > 0)); then
        printf '  %s\n' "${COMPLEX_RUN_MATS[@]}"
    else
        echo "  (none)"
    fi
    echo "spd matrices (${#SELECTED_SPD[@]}):"
    if ((${#SELECTED_SPD[@]} > 0)); then
        printf '  %s\n' "${SELECTED_SPD[@]}"
    else
        echo "  (none)"
    fi
    echo
    cargo run -q --release --manifest-path "$RUSSELL_DIR/Cargo.toml" \
        --package russell_sparse "${FEATURE_ARGS[@]}" --bin sys_info
} >"$INFO_FILE"
echo "system information written to $INFO_FILE" >&2

run_matrix() {
    local name=$1
    CURRENT_MAT=$((CURRENT_MAT + 1))
    local file="$INPUT_DIR/$name.mtx"
    local out="$SOLVER_DIR/$name.json"
    if [[ ! -f "$file" ]]; then
        echo "skip $CURRENT_MAT/$TOTAL_MATS $name: $file not found" >&2
        return
    fi

    local spd_label=""
    local run_args=("${ARGS[@]}")
    if is_spd "$name"; then
        spd_label=" (spd)"
        run_args+=(--positive-definite)
    fi

    if ((INFORMATION)); then
        echo "=== $CURRENT_MAT/$TOTAL_MATS $name$spd_label (information only) ===" >&2
    else
        echo "=== $CURRENT_MAT/$TOTAL_MATS $name$spd_label ===" >&2
    fi
    # A failed solve is a result, not a script failure; the JSON records it.
    if ! cargo run -q --release --manifest-path "$RUSSELL_DIR/Cargo.toml" \
        --package russell_sparse "${FEATURE_ARGS[@]}" \
        --bin solve_matrix_market -- "${run_args[@]}" "$file" >"$out"; then
        # Do not leave an empty result behind when the program failed before writing JSON.
        [[ -s "$out" ]] || rm -f "$out"
    fi
}

TIMEFORMAT='total time: %3R s'
time {
    for SOLVER in "${GENIES[@]}"; do
        SOLVER_DIR="$OUTPUT_DIR/$SOLVER"
        mkdir -p "$SOLVER_DIR"
        ARGS=(-g "$SOLVER" --nrun "$NRUN" --time-guard "$TIME_GUARD")
        if ((INFORMATION)); then
            ARGS+=(--information)
        fi
        if ((ALL_GENIES)); then
            echo "=== Genie: $SOLVER ===" >&2
        fi

        SOLVER_MATS=("${REAL_RUN_MATS[@]}" "${COMPLEX_RUN_MATS[@]}")

        TOTAL_MATS=${#SOLVER_MATS[@]}
        CURRENT_MAT=0
        for name in "${SOLVER_MATS[@]}"; do
            run_matrix "$name"
        done
    done
}

if ((ALL_GENIES)); then
    echo "results written to $OUTPUT_DIR" >&2
else
    echo "results written to $SOLVER_DIR" >&2
fi
