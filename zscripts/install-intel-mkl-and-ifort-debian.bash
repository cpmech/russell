#!/bin/bash

set -e

VERSION="latest"

# fake sudo function to be used by docker build
sudo () {
  [[ $EUID = 0 ]] || set -- command sudo "$@"
  "$@"
}

# update sources.list
wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null
sudo echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list

# install packages
sudo apt-get update -y && \
sudo apt-get install -y --no-install-recommends \
    intel-oneapi-mkl \
    intel-oneapi-mkl-devel

# TODO:
# intel-oneapi-compiler-fortran-$VERSION

LIBDIR1="/opt/intel/oneapi/mkl/$VERSION/lib/intel64"
LIBDIR2="/opt/intel/oneapi/compiler/$VERSION/lib"

# remove this strange file that comes with MKL
WEIRD_FILE="/opt/intel/oneapi/mkl/$VERSION/lib/intel64/libmkl_sycl.so"
if [ -f "$WEIRD_FILE" ]; then
    sudo mv "$WEIRD_FILE" "/opt/intel/oneapi/mkl/$VERSION/lib/intel64/libmkl_sick.txt"
fi

# update ldconfig
echo "${LIBDIR1}" | sudo tee /etc/ld.so.conf.d/intel-oneapi-mkl-and-compiler.conf >/dev/null
echo "${LIBDIR2}" | sudo tee -a /etc/ld.so.conf.d/intel-oneapi-mkl-and-compiler.conf >/dev/null
sudo ldconfig
