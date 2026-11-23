#!/bin/bash

# Exit on error (-e), treat unset variables as errors (-u), and propagate
# pipeline failures (-o pipefail) so any silent failure is caught early
set -euo pipefail

# Configuration
PREFIX="/ucrt64"
INCDIR="$PREFIX/include/mumps"
LIBDIR="$PREFIX/lib/mumps"
PDIR="$(pwd)/zscripts/makefiles-mumps"

# Install build tools and BLAS/LAPACK dependencies;
pacman -S --noconfirm base-devel curl git \
    mingw-w64-ucrt-x86_64-gcc-fortran \
    mingw-w64-ucrt-x86_64-make \
    mingw-w64-ucrt-x86_64-cmake \
    mingw-w64-ucrt-x86_64-clang \
    mingw-w64-ucrt-x86_64-metis

# The MUMPS source is the thread-safe fork: every MUMPS call is dispatched to a
# persistent worker inside the library, so clients need no lock or worker of their own.
MUMPS_VERSION="5.9.2"
MUMPS_DIR="MUMPS_${MUMPS_VERSION//./}"
MUMPS_REPO="https://github.com/cpmech/${MUMPS_DIR}.git"

cd /tmp
rm -rf "$MUMPS_DIR"
git clone -q --depth 1 "$MUMPS_REPO" "$MUMPS_DIR"
echo "... MUMPS source: $MUMPS_REPO main ($(git -C "$MUMPS_DIR" rev-parse --short HEAD))"
cd "$MUMPS_DIR"

# Copy the Makefile.inc file
cp "$PDIR/Makefile.inc.msys2" Makefile.inc

# Create installation directories
mkdir -p "$LIBDIR" "$INCDIR"

# Compile the double precision (real) version
make d
cp -av lib/lib*.a "$LIBDIR/"

# Compile the complex version
make clean
make z
cp -av lib/lib*.a "$LIBDIR/"

# Install the public headers
cp -av include/*.h "$INCDIR/"
