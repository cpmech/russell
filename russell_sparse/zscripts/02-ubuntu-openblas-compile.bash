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
    gfortran \
    git \
    libmetis-dev \
    make \
    patch

# compile and install MUMPS
bash zscripts/install-mumps.bash

# compile and install UMFPACK
bash zscripts/install-umfpack.bash
