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
# or 0 (default) to use the OpenBLAS toolchain.
# When using MKL, run debian-install-intel-toolkit.bash first.
USE_INTEL_MKL=${1:-0}

# Installation paths
PREFIX="/usr/local"
INCDIR=$PREFIX/include/suitesparse
LIBDIR=$PREFIX/lib/suitesparse

# Install build tools and BLAS/LAPACK dependencies
sudo apt-get update -y
sudo apt-get install -y --no-install-recommends cmake curl git make
if [ "${USE_INTEL_MKL}" = "1" ]; then
    set +u  # setvars.sh references variables before setting them; suppress -u temporarily
    source /opt/intel/oneapi/setvars.sh
    set -u
    export | grep -i MKLROOT
else
    sudo apt-get install -y --no-install-recommends clang liblapacke-dev libopenblas-dev
fi

# set the compiler/linker and cmake options for SuiteSparse
if [ "${USE_INTEL_MKL}" = "1" ]; then
    COMP_LINK="-DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx"
    CMAKE_OPTIONS="${COMP_LINK} -DBLA_VENDOR=Intel10_64lp -DBLA_SIZEOF_INTEGER=4 -DSUITESPARSE_USE_FORTRAN=OFF"
else
    COMP_LINK="-DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++"
    CMAKE_OPTIONS="${COMP_LINK} -DBLA_VENDOR=OpenBLAS -DBLA_SIZEOF_INTEGER=4 -DSUITESPARSE_USE_FORTRAN=OFF"
fi

# download the source code
cd /tmp
if ! [ -d "SuiteSparse" ]; then
    git clone https://github.com/DrTimothyAldenDavis/SuiteSparse.git
fi
cd SuiteSparse

# function to compile and copy results to local directories
action () {
    local dir=$1
    cd $dir
    make clean
    CMAKE_OPTIONS=${CMAKE_OPTIONS} make local
    CMAKE_OPTIONS=${CMAKE_OPTIONS} make install
    cd ..
}

# compile and copy results to local directories
action SuiteSparse_config
action AMD
action CAMD
action CCOLAMD
action COLAMD
action CHOLMOD
action UMFPACK
action BTF
action KLU

# copy include files
sudo mkdir -p $INCDIR/
sudo cp -av include/suitesparse/*.h $INCDIR/

# copy libray files
sudo mkdir -p $LIBDIR/
sudo cp -av lib/* $LIBDIR/

# update ldconfig
echo "${LIBDIR}" | sudo tee /etc/ld.so.conf.d/suitesparse.conf >/dev/null
sudo ldconfig
