#!/bin/bash

set -e

# fake sudo function to be used by docker build
sudo () {
  [[ $EUID = 0 ]] || set -- command sudo "$@"
  "$@"
}

# install dependencies
sudo apt-get update -y && \
sudo apt-get install -y --no-install-recommends \
    cmake \
    g++ \
    gdb \
    git \
    libmetis-dev \
    make \
    patch

# install Intel MKL
bash zscripts/install-intel-mkl-and-ifort-debian.bash

# compile and install MUMPS
bash zscripts/compile-and-install-mumps mkl

# compile and install UMFPACK
bash zscripts/compile-and-install-umfpack mkl
