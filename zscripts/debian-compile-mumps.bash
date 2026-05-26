#!/bin/bash

# Exit on error (-e), treat unset variables as errors (-u), and propagate
# pipeline failures (-o pipefail) so any silent failure is caught early
set -euo pipefail

# When running as root (e.g. inside Docker), act as a no-op wrapper;
# otherwise delegate to the real sudo
sudo () {
  [[ $EUID = 0 ]] || set -- command sudo "$@"
  "$@"
}

# Optional first argument: set to 1 to build against Intel MKL/oneAPI,
# or 0 (default) to use the OpenBLAS + gfortran toolchain.
# When using MKL, run debian-install-intel-mkl.bash first.
USE_INTEL_MKL=${1:-0}

# Installation paths and source version
VERSION="5.9.0"
PREFIX="/usr/local"
INCDIR="$PREFIX/include/mumps"
LIBDIR="$PREFIX/lib/mumps"
PDIR="$(pwd)/zscripts/makefiles-mumps"

# Install build tools and BLAS/LAPACK dependencies;
# libmetis-dev is needed for graph-partitioning-based reordering
sudo apt-get update -y
sudo apt-get install -y --no-install-recommends curl libmetis-dev make
if [ "${USE_INTEL_MKL}" = "1" ]; then
    set +u  # setvars.sh references variables before setting them; suppress -u temporarily
    source /opt/intel/oneapi/setvars.sh  # sets CC, FC, and PATH for the ifx compiler
    set -u
    export | grep -i MKLROOT
else
    sudo apt-get install -y --no-install-recommends gfortran liblapacke-dev libopenblas-dev
fi

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

# Select the Makefile.inc that matches the chosen BLAS/LAPACK backend
if [ "${USE_INTEL_MKL}" = "1" ]; then
    cp "$PDIR/MakefileMKL.inc" Makefile.inc
else
    cp "$PDIR/Makefile.inc" Makefile.inc
fi

# Create installation directories
sudo mkdir -p "$LIBDIR" "$INCDIR"

# Build all four variants and copy the resulting libraries;
# make clean between targets is required because object files are not
# recompiled when only the output target changes
make d                              # double-precision, static
sudo cp -av lib/lib*.a "$LIBDIR/"

make clean
make dshared                        # double-precision, shared
sudo cp -av lib/lib*.so "$LIBDIR/"

make clean
make z                              # complex (double), static
sudo cp -av lib/lib*.a "$LIBDIR/"

make clean
make zshared                        # complex (double), shared
sudo cp -av lib/lib*.so "$LIBDIR/"

# Install public headers
sudo cp -av include/*.h "$INCDIR/"

# Register the new library path with the dynamic linker
echo "$LIBDIR" | sudo tee /etc/ld.so.conf.d/mumps.conf >/dev/null
sudo ldconfig
