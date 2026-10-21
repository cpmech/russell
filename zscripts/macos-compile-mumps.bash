#!/bin/bash
# Compiles and installs the sequential MUMPS solver on macOS.
#
# The libraries come from the Debian source tarball, are built with Homebrew's
# gfortran against Apple's Accelerate framework, and carry the project's
# `_cpmech` suffix (`PLAT` in the makefile) so they cannot be confused with a
# system MUMPS. Install names are rewritten to `@rpath/...`, which is what macOS
# needs for the run path that the crate's build script adds to resolve the whole
# library family.
#
# Usage: macos-compile-mumps.bash [PREFIX]      (default: /usr/local)
#
# gfortran must be available (brew install gcc).
set -euo pipefail

PREFIX="${1:-/usr/local}"
INCDIR="$PREFIX/include/mumps"
LIBDIR="$PREFIX/lib/mumps"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PDIR="$HERE/makefiles-mumps"
VERSION="5.9.0"

if ! command -v gfortran >/dev/null 2>&1; then
    echo "macos-compile-mumps: gfortran is required (brew install gcc)" >&2
    exit 1
fi

# Run a command with sudo only when the destination is not writable.
as_root() {
    if [ -w "$PREFIX" ] || [ -w "$(dirname "$PREFIX")" ]; then
        "$@"
    else
        sudo "$@"
    fi
}

MUMPS_GZ="mumps_${VERSION}.orig.tar.gz"
MUMPS_DIR="MUMPS_${VERSION}"
cd /tmp
if [ -d "$MUMPS_DIR" ]; then
    echo "... removing previous $MUMPS_DIR directory"
    rm -rf "$MUMPS_DIR"
fi
if [ -f "$MUMPS_GZ" ]; then
    echo "... using existing $MUMPS_GZ file"
else
    curl -fL "http://deb.debian.org/debian/pool/main/m/mumps/${MUMPS_GZ}" -o "$MUMPS_GZ"
fi

tar xzf "$MUMPS_GZ"
cd "$MUMPS_DIR"
cp "$PDIR/Makefile.inc.macos" Makefile.inc

# Only the shared double and complex libraries are needed: the crate links
# dmumps_cpmech and zmumps_cpmech, and their own dependencies are recorded in
# each dylib.
make dshared zshared

as_root mkdir -p "$LIBDIR" "$INCDIR"
as_root cp -av lib/lib*.dylib "$LIBDIR/"
as_root cp -av include/*.h "$INCDIR/"

# Point every library at @rpath/<name>; the build script puts the library
# directory on the run path of this package's binaries.
cd "$LIBDIR"
for f in lib*.dylib; do
    install_name_tool -id "@rpath/$f" "$f"
done
for f in lib*.dylib; do
    for dep in lib*.dylib; do
        install_name_tool -change "$dep" "@rpath/$dep" "$f" 2>/dev/null || true
        install_name_tool -change "/tmp/$MUMPS_DIR/lib/$dep" "@rpath/$dep" "$f" 2>/dev/null || true
        install_name_tool -change "$LIBDIR/$dep" "@rpath/$dep" "$f" 2>/dev/null || true
    done
done

echo "... installed into $LIBDIR"
