#!/bin/bash

set -e

VERSION="2023.2.0"

# install packages
sudo apt-get remove \
    intel-oneapi-compiler-fortran-$VERSION \
    intel-oneapi-mkl-$VERSION \
    intel-oneapi-mkl-devel-$VERSION

# update ldconfig
sudo rm /etc/ld.so.conf.d/intel-oneapi-mkl-and-compiler.conf
sudo ldconfig
