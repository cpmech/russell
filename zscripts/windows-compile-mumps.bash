#!/bin/bash

# Exit on error (-e), treat unset variables as errors (-u), and propagate
# pipeline failures (-o pipefail) so any silent failure is caught early
set -euo pipefail

# Configuration
VERSION="5.9.0"
PREFIX="/ucrt64"
INCDIR="$PREFIX/include/mumps"
LIBDIR="$PREFIX/lib/mumps"
PDIR="$(pwd)/zscripts/makefiles-mumps"

# Install build tools and BLAS/LAPACK dependencies;
pacman -S --noconfirm base-devel curl \
    mingw-w64-ucrt-x86_64-gcc-fortran \
    mingw-w64-ucrt-x86_64-make \
    mingw-w64-ucrt-x86_64-cmake \
    mingw-w64-ucrt-x86_64-clang \
    mingw-w64-ucrt-x86_64-metis

# Download the source tarball from Debian (reuse it if already present)
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
