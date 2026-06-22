#!/bin/bash

# Update the library version constants in src/util.rs by reading the
# version variables from the project's installation scripts.
# Run from the repository root.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UTIL_RS="$ROOT/russell_sparse/src/util.rs"

error() { echo "ERROR: $*" >&2; exit 1; }

[[ -f "$UTIL_RS" ]] || error "util.rs not found at $UTIL_RS"

SCRIPT_CUDSS="$ROOT/zscripts/linux-compile-cudss.bash"
SCRIPT_MUMPS="$ROOT/zscripts/debian-compile-mumps.bash"

[[ -f "$SCRIPT_CUDSS" ]] || error "cuDSS install script not found"
[[ -f "$SCRIPT_MUMPS" ]] || error "MUMPS install script not found"

# Extract version variables from the install scripts
CUDSS_VERSION=$(grep -oP '^CUDSS_VERSION="\K[^"]+' "$SCRIPT_CUDSS" | head -1)
CUDA_VERSION=$(  grep -oP '^CUDA_VERSION="\K[^"]+'   "$SCRIPT_CUDSS" | head -1)
MUMPS_VERSION=$( grep -oP '^VERSION="\K[^"]+'        "$SCRIPT_MUMPS" | head -1)

[[ -n "$CUDSS_VERSION" ]] || error "could not parse CUDSS_VERSION from $SCRIPT_CUDSS"
[[ -n "$CUDA_VERSION"   ]] || error "could not parse CUDA_VERSION from $SCRIPT_CUDSS"
[[ -n "$MUMPS_VERSION"  ]] || error "could not parse MUMPS_VERSION from $SCRIPT_MUMPS"

echo "Detected versions:"
echo "  cuDSS  = $CUDSS_VERSION (CUDA $CUDA_VERSION)"
echo "  MUMPS  = $MUMPS_VERSION"
echo ""

# Replace the constants in util.rs
sed -i \
    -e "s/^const CUDSS_SCRIPT_VERSION: &str = \".*\";/const CUDSS_SCRIPT_VERSION: \&str = \"${CUDSS_VERSION}\";/" \
    -e "s/^const CUDA_SCRIPT_VERSION: &str = \".*\";/const CUDA_SCRIPT_VERSION: \&str = \"${CUDA_VERSION}\";/" \
    -e "s/^const MUMPS_SCRIPT_VERSION: &str = \".*\";/const MUMPS_SCRIPT_VERSION: \&str = \"${MUMPS_VERSION}\";/" \
    "$UTIL_RS"

echo "Updated $UTIL_RS"
