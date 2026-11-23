#!/bin/bash

# Update the library version constants in src/library_info.rs by reading the
# version variables from the project's installation scripts.
# Run from the repository root.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIBRARY_INFO_RS="$ROOT/russell_sparse/src/library_info.rs"

error() { echo "ERROR: $*" >&2; exit 1; }

[[ -f "$LIBRARY_INFO_RS" ]] || error "library_info.rs not found at $LIBRARY_INFO_RS"

SCRIPT_MUMPS="$ROOT/zscripts/debian-compile-mumps.bash"

[[ -f "$SCRIPT_MUMPS" ]] || error "MUMPS install script not found"

# Extract version variables from the install scripts
MUMPS_VERSION=$(grep -oP '^MUMPS_VERSION="\K[^"]+' "$SCRIPT_MUMPS" | head -1 || true)

[[ -n "$MUMPS_VERSION"  ]] || error "could not parse MUMPS_VERSION from $SCRIPT_MUMPS"

echo "Detected versions:"
echo "  MUMPS  = $MUMPS_VERSION"
echo ""

# Replace the constants in library_info.rs
sed -i \
    -e "s/^const MUMPS_SCRIPT_VERSION: &str = \".*\";/const MUMPS_SCRIPT_VERSION: \&str = \"${MUMPS_VERSION}\";/" \
    "$LIBRARY_INFO_RS"

echo "Updated $LIBRARY_INFO_RS"
