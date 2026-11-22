#!/bin/bash

set -euo pipefail

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

# Symmetric positive-definite matrices
#
# Source: the Sym column of the archived arch-performance-tables.pdf (Yes*),
# plus pres-cylin-3d-tet10-coarse, whose QDLDL positive inertia equals its
# dimension. These get the solver option --positive-definite (-p).
SPD_MATS=(
    pmsim-hex8-27
    oilpan
    inline_1
    PFlow_742
    Emilia_923
    Flan_1565
    pres-cylin-3d-tet10-coarse
    pres-cylin-3d-tet10-fine
)

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

# Symmetric matrices
#
# Source: the Sym column of the archived arch-performance-tables.pdf.
# QDLDL requires a symmetric matrix; the other direct solvers do not.
SYMMETRIC_MATS=(
    boyd2
    darcy003
    dielFilterV2real
    Emilia_923
    Flan_1565
    helm2d03
    inline_1
    oilpan
    PFlow_742
    pmsim-hex8-27
    pres-cylin-3d-tet10-coarse
    pres-cylin-3d-tet10-fine
    mplate
    dielFilterV3clx
)

# Returns "matrix" or "matrices" for a count
matrix_word() {
    if (($1 == 1)); then
        echo "matrix"
    else
        echo "matrices"
    fi
}

# Returns true when the matrix is symmetric
is_symmetric() {
    local name=$1
    local candidate
    for candidate in "${SYMMETRIC_MATS[@]}"; do
        if [[ "$candidate" == "$name" ]]; then
            return 0
        fi
    done
    return 1
}

# Returns true for iterative solvers
is_iterative() {
    local solver
    solver=$(printf '%s' "$1" | tr '[:lower:]' '[:upper:]')
    case "$solver" in
        CG | BICGSTAB | GMRES) return 0 ;;
        *) return 1 ;;
    esac
}

# Returns true for solvers without a complex-valued implementation
is_real_only() {
    local solver
    solver=$(printf '%s' "$1" | tr '[:lower:]' '[:upper:]')
    case "$solver" in
        CG | BICGSTAB | GMRES | BASICLU | QDLDL) return 0 ;;
        *) return 1 ;;
    esac
}

# Returns true for solvers that require a symmetric matrix
is_symmetric_only() {
    local solver
    solver=$(printf '%s' "$1" | tr '[:lower:]' '[:upper:]')
    case "$solver" in
        QDLDL) return 0 ;;
        *) return 1 ;;
    esac
}

# Returns true for solvers that run single-threaded by default
is_single_threaded() {
    local solver
    solver=$(printf '%s' "$1" | tr '[:lower:]' '[:upper:]')
    case "$solver" in
        FAER | BASICLU | QDLDL | UMFPACK | CG | BICGSTAB | GMRES) return 0 ;;
        *) return 1 ;;
    esac
}

HERE=$(cd "$(dirname "$0")" && pwd)
RUSSELL_DIR=${RUSSELL_DIR:-$(cd "$HERE/../../.." && pwd)}
INPUT_DIR=${INPUT_DIR:-$HOME/Downloads/matrix-market}
SYSTEM=${SYSTEM:-arch}
OUTPUT_DIR=${OUTPUT_DIR:-/tmp/russell_sparse/results/$SYSTEM}
FEATURES=${FEATURES-intel_mkl,umfpack,mumps}
SOLVER=FAER

# Pin single-threaded solvers to the P-cores on hybrid CPUs
PIN_SINGLE_THREAD=${PIN_SINGLE_THREAD:-1}
P_CORES=${P_CORES:-}
if ((PIN_SINGLE_THREAD)); then
    if ! command -v taskset >/dev/null 2>&1; then
        echo "warning: taskset not found; single-threaded solvers will not be pinned" >&2
        PIN_SINGLE_THREAD=0
    elif [[ -z "$P_CORES" && -r /sys/devices/cpu_core/cpus ]]; then
        P_CORES=$(cat /sys/devices/cpu_core/cpus)
    fi
    if [[ -z "$P_CORES" ]]; then
        echo "warning: P-core list not found; single-threaded solvers will not be pinned" >&2
        PIN_SINGLE_THREAD=0
    fi
fi

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
    echo "  Iterative solvers and BasicLU/QDLDL skip the complex matrices."
    echo "  QDLDL also skips non-symmetric matrices."
    echo ""
    echo "Environment: RUSSELL_DIR, INPUT_DIR, OUTPUT_DIR, INFO_FILE, SYSTEM, FEATURES, PIN_SINGLE_THREAD, P_CORES"
}

INFORMATION=1
ALL_GENIES=0
FIRST_N=0
FIRST_COMPLEX_N=0
NRUN=1
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

if ((ALL_GENIES)); then
    GENIES=()
    if [[ "$FEATURES" == *mumps* ]]; then
        GENIES+=(mumps)
    fi
    if [[ "$FEATURES" == *umfpack* ]]; then
        GENIES+=(umfpack)
    fi
    GENIES+=(faer qdldl basiclu cg bicgstab gmres)
else
    shopt -s nocasematch
    if [[ ! "$SOLVER" =~ ^(faer|cg|bicgstab|gmres|mumps|umfpack|basiclu|qdldl)$ ]]; then
        echo "invalid solver: $SOLVER (choose one of: faer, cg, bicgstab, gmres, mumps, umfpack, basiclu, qdldl)" >&2
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

echo "=== Building solve_matrix_market and sys_info (features: ${FEATURES:-none}) ===" >&2
cargo build --release --manifest-path "$RUSSELL_DIR/Cargo.toml" \
    --package russell_sparse "${FEATURE_ARGS[@]}" --bin solve_matrix_market --bin sys_info

{
    echo "Benchmark configuration"
    echo "date: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    echo "system: $SYSTEM"
    echo "features: ${FEATURES:-none}"
    echo "solvers: ${GENIES[*]}"
    echo "runs: $NRUN"
    echo "input: $INPUT_DIR"
    echo "output: $OUTPUT_DIR"
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

    local pin=()
    if ((PIN_SINGLE_THREAD)) && ((!INFORMATION)) && is_single_threaded "$SOLVER"; then
        pin=(env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c "$P_CORES")
    fi

    if ((INFORMATION)); then
        echo "=== $CURRENT_MAT/$TOTAL_MATS $name$spd_label (information only) ===" >&2
    else
        echo "=== $CURRENT_MAT/$TOTAL_MATS $name$spd_label ===" >&2
    fi
    # A failed solve is a result, not a script failure; the JSON records it.
    if ! "${pin[@]}" cargo run -q --release --manifest-path "$RUSSELL_DIR/Cargo.toml" \
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
        ARGS=(-g "$SOLVER" --nrun "$NRUN")
        if ((INFORMATION)); then
            ARGS+=(--information)
        fi
        if ((ALL_GENIES)); then
            echo "=== Genie: $SOLVER ===" >&2
        fi

        # Select the matrices supported by this solver
        if is_real_only "$SOLVER"; then
            CANDIDATE_MATS=("${REAL_RUN_MATS[@]}")
            if ((${#COMPLEX_RUN_MATS[@]} > 0)); then
                echo "=== Skipping ${#COMPLEX_RUN_MATS[@]} complex $(matrix_word "${#COMPLEX_RUN_MATS[@]}") (solver $SOLVER) ===" >&2
            fi
        else
            CANDIDATE_MATS=("${REAL_RUN_MATS[@]}" "${COMPLEX_RUN_MATS[@]}")
        fi

        SOLVER_MATS=()
        if is_symmetric_only "$SOLVER"; then
            SKIPPED_MATS=0
            for name in "${CANDIDATE_MATS[@]}"; do
                if is_symmetric "$name"; then
                    SOLVER_MATS+=("$name")
                else
                    SKIPPED_MATS=$((SKIPPED_MATS + 1))
                fi
            done
            if ((SKIPPED_MATS > 0)); then
                echo "=== Skipping $SKIPPED_MATS non-symmetric $(matrix_word "$SKIPPED_MATS") (symmetric-only solver $SOLVER) ===" >&2
            fi
        else
            SOLVER_MATS=("${CANDIDATE_MATS[@]}")
        fi

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
