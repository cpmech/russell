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

# Replaces the home directory prefix with ~ for progress and provenance output
display_path() {
    local path=$1
    local home=${HOME:-}
    if [[ -n "$home" ]]; then
        if [[ "$path" == "$home" ]]; then
            echo "~"
            return
        fi
        if [[ "$path" == "$home"/* ]]; then
            echo "~${path#"$home"}"
            return
        fi
    fi
    echo "$path"
}

HERE=$(cd "$(dirname "$0")" && pwd)
RUSSELL_DIR=${RUSSELL_DIR:-$(cd "$HERE/../../.." && pwd)}
INPUT_DIR=${INPUT_DIR:-$HOME/Downloads/matrix-market}
SYSTEM=${SYSTEM:-arch}
OUTPUT_DIR=${OUTPUT_DIR:-/tmp/russell_sparse/results/$SYSTEM}
FEATURES=${FEATURES-intel_mkl,umfpack,mumps}
MATRICES_JSON=${MATRICES_JSON:-"$RUSSELL_DIR/russell_sparse/data/matrix_market/matrices.json"}
SOLVER=FAER

# The CPU topology and the thread counts come from `machine_info --cpu-plan`
# once the binary has been built below. The Rust plan also honors P_CORES,
# E_CORES, MUMPS_THREADS, and BLAS_THREADS from the environment.
CPU_MODEL=$(sed -n 's/^model name[[:space:]]*: //p' /proc/cpuinfo 2>/dev/null | head -n 1)
CPU_MODEL=${CPU_MODEL:-unknown}

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
    echo "Environment: RUSSELL_DIR, INPUT_DIR, OUTPUT_DIR, INFO_FILE, MATRICES_JSON, SYSTEM, FEATURES,"
    echo "             TIME_GUARD, P_CORES, E_CORES, MUMPS_THREADS, BLAS_THREADS"
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

echo "=== Building solve_matrix_market and machine_info (features: ${FEATURES:-none}) ===" >&2
cargo build --release --manifest-path "$RUSSELL_DIR/Cargo.toml" \
    --package russell_sparse "${FEATURE_ARGS[@]}" --bin solve_matrix_market --bin machine_info

# The CPU plan is computed by the Rust binary: P/E cores, physical counts, the
# MUMPS/BLAS thread counts, and the P-core affinity list. The same environment
# overrides the previous shell implementation supported are still honored.
PLAN=$(cargo run -q --release --manifest-path "$RUSSELL_DIR/Cargo.toml" \
    --package russell_sparse "${FEATURE_ARGS[@]}" --bin machine_info -- --cpu-plan)
P_CORES=""
E_CORES=""
P_LOGICAL=0
E_LOGICAL=0
P_PHYSICAL=0
E_PHYSICAL=0
MUMPS_THREADS=0
BLAS_THREADS=0
CPU_AFFINITY="none"
while IFS='=' read -r key value; do
    case "$key" in
        P_CORES) P_CORES=$value ;;
        E_CORES) E_CORES=$value ;;
        P_LOGICAL) P_LOGICAL=$value ;;
        E_LOGICAL) E_LOGICAL=$value ;;
        P_PHYSICAL) P_PHYSICAL=$value ;;
        E_PHYSICAL) E_PHYSICAL=$value ;;
        MUMPS_THREADS) MUMPS_THREADS=$value ;;
        BLAS_THREADS) BLAS_THREADS=$value ;;
        CPU_AFFINITY) CPU_AFFINITY=$value ;;
    esac
done <<<"$PLAN"

if [[ "$(uname -s)" != "Linux" ]]; then
    CPU_AFFINITY="none"
fi

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
    echo "cpu_model: $CPU_MODEL"
    echo "p_cores: ${P_CORES:-none}"
    echo "e_cores: ${E_CORES:-none}"
    echo "p_logical: $P_LOGICAL"
    echo "e_logical: $E_LOGICAL"
    echo "p_physical: $P_PHYSICAL"
    echo "e_physical: $E_PHYSICAL"
    echo "mumps_threads: $MUMPS_THREADS"
    echo "blas_threads: $BLAS_THREADS"
    echo "cpu_affinity: $CPU_AFFINITY"
    echo "input: $(display_path "$INPUT_DIR")"
    echo "output: $(display_path "$OUTPUT_DIR")"
    echo
    echo "solver arguments:"
    for solver in "${GENIES[@]}"; do
        printf '  -g %s --nrun %s --time-guard %s' "$solver" "$NRUN" "$TIME_GUARD"
        solver_key=$(printf '%s' "$solver" | tr '[:upper:]' '[:lower:]')
        if [[ "$solver_key" == "mumps" ]]; then
            printf ' --nt %s --mumps-nt %s' "$MUMPS_THREADS" "$MUMPS_THREADS"
        else
            printf ' --nt %s' "$BLAS_THREADS"
        fi
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
        --package russell_sparse "${FEATURE_ARGS[@]}" --bin machine_info
} >"$INFO_FILE"
echo "machine information written to $(display_path "$INFO_FILE")" >&2

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
        solver_key=$(printf '%s' "$SOLVER" | tr '[:upper:]' '[:lower:]')
        if [[ "$solver_key" == "mumps" ]]; then
            ARGS+=(--nt "$MUMPS_THREADS" --mumps-nt "$MUMPS_THREADS")
        else
            ARGS+=(--nt "$BLAS_THREADS")
        fi
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
