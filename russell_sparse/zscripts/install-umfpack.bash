#!/bin/bash

set -e

# first argument
BLAS_LIB=${1:-""}

# options
PREFIX="/usr/local"
INCDIR=$PREFIX/include/umfpack
LIBDIR=$PREFIX/lib/umfpack

# fake sudo function to be used by docker build
sudo () {
  [[ $EUID = 0 ]] || set -- command sudo "$@"
  "$@"
}

# BLAS lib
CMAKE_OPTIONS="-DBLA_VENDOR=OpenBLAS -DBLA_SIZEOF_INTEGER=4 -DNFORTRAN=ON"
if [ "${BLAS_LIB}" = "mkl" ]; then
    CMAKE_OPTIONS="-DBLA_VENDOR=Intel10_64lp -DBLA_SIZEOF_INTEGER=4 -DNFORTRAN=ON"
    source /opt/intel/oneapi/setvars.sh
    export | grep -i MKLROOT
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

# copy include files
sudo mkdir -p $INCDIR/
sudo cp -av include/*.h $INCDIR/

# copy libray files
sudo mkdir -p $LIBDIR/
sudo cp -av lib/* /usr/local/lib/umfpack

# update ldconfig
echo "${LIBDIR}" | sudo tee /etc/ld.so.conf.d/umfpack.conf >/dev/null
sudo ldconfig
