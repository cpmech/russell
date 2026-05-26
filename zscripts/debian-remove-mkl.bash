#!/bin/bash

set -e

# install packages
sudo apt-get remove \
    intel-oneapi-compiler-fortran \
    intel-oneapi-mkl \
    intel-oneapi-mkl-devel

# update ldconfig
if [ -f "/etc/ld.so.conf.d/intel-oneapi-mkl-and-compiler.conf" ]; then
    sudo rm /etc/ld.so.conf.d/intel-oneapi-mkl-and-compiler.conf
    sudo ldconfig
fi
