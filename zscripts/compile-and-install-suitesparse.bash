#!/bin/bash

set -e

# fake sudo function to be used by docker build
sudo () {
  [[ $EUID = 0 ]] || set -- command sudo "$@"
  "$@"
}

# the first argument is the "mkl" option
BLAS_LIB=${1:-""}

# options
PREFIX="/usr/local"
INCDIR=$PREFIX/include/suitesparse
LIBDIR=$PREFIX/lib/suitesparse

# install dependencies
sudo apt-get update -y &&
sudo apt-get install -y --no-install-recommends \
    cmake \
    curl \
    g++ \
    git \
    make
if [ "${BLAS_LIB}" = "mkl" ]; then
    bash zscripts/install-intel-mkl-and-ifort-debian.bash
else
    sudo apt-get install -y --no-install-recommends \
        liblapacke-dev \
        libopenblas-dev
fi

# source Intel oneAPI vars (ifort)
if [ "${BLAS_LIB}" = "mkl" ]; then
    source /opt/intel/oneapi/setvars.sh
    export | grep -i MKLROOT
fi

# set cmake options
CMAKE_OPTIONS="-DBLA_VENDOR=OpenBLAS -DBLA_SIZEOF_INTEGER=4 -DNFORTRAN=ON"
if [ "${BLAS_LIB}" = "mkl" ]; then
    CMAKE_OPTIONS="-DBLA_VENDOR=Intel10_64lp -DBLA_SIZEOF_INTEGER=4 -DNFORTRAN=ON"
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
sudo cp -av lib/* /usr/local/lib/suitesparse

# update ldconfig
echo "${LIBDIR}" | sudo tee /etc/ld.so.conf.d/suitesparse.conf >/dev/null
sudo ldconfig
