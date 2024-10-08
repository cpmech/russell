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
VERSION="5.7.3"
PREFIX="/usr/local"
INCDIR=$PREFIX/include/mumps
LIBDIR=$PREFIX/lib/mumps
PDIR=`pwd`/zscripts/makefiles-mumps

# install dependencies
sudo apt-get update -y &&
sudo apt-get install -y --no-install-recommends \
    cmake \
    curl \
    g++ \
    libmetis-dev \
    make
if [ "${BLAS_LIB}" = "mkl" ]; then
    bash zscripts/install-intel-mkl-and-ifort-debian.bash
else
    sudo apt-get install -y --no-install-recommends \
        gfortran \
        liblapacke-dev \
        libopenblas-dev
fi

# source Intel oneAPI vars (ifort)
if [ "${BLAS_LIB}" = "mkl" ]; then
    source /opt/intel/oneapi/setvars.sh
fi

# download the source code
MUMPS_GZ=mumps_$VERSION.orig.tar.gz
MUMPS_DIR=MUMPS_$VERSION
cd /tmp
if [ -d "$MUMPS_DIR" ]; then
    echo "... removing previous $MUMPS_DIR directory"
    rm -rf $MUMPS_DIR
fi
if [ -f "$MUMPS_GZ" ]; then
    echo "... using existing $MUMPS_GZ file"
else
    curl http://deb.debian.org/debian/pool/main/m/mumps/$MUMPS_GZ -o $MUMPS_GZ
fi

# extract the source code into /tmp dir
tar xzf $MUMPS_GZ
cd $MUMPS_DIR

# copy inc file
if [ "${BLAS_LIB}" = "mkl" ]; then
    cp $PDIR/MakefileMKL.inc Makefile.inc
else
    cp $PDIR/Makefile.inc Makefile.inc
fi

# create output lib dir
sudo mkdir -p $LIBDIR/

# compile double (static)
make d
sudo cp -av lib/lib*.a $LIBDIR/

# compile double (shared)
make clean # << important
make dshared
sudo cp -av lib/lib*.so $LIBDIR/

# compile complex (static)
make clean # << important
make z
sudo cp -av lib/lib*.a $LIBDIR/

# compile complex (shared)
make clean # << important
make zshared
sudo cp -av lib/lib*.so $LIBDIR/

# copy include files
sudo mkdir -p $INCDIR/
sudo cp -av include/*.h $INCDIR/

# update ldconfig
echo "${LIBDIR}" | sudo tee /etc/ld.so.conf.d/mumps.conf >/dev/null
sudo ldconfig
