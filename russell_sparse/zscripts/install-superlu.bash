#!/bin/bash

set -e

# fake sudo function to be used by docker build
sudo () {
  [[ $EUID = 0 ]] || set -- command sudo "$@"
  "$@"
}

# options
PREFIX="/usr/local"
INCDIR=$PREFIX/include/superlu
LIBDIR=$PREFIX/lib/superlu

# download the source code
cd /tmp
if ! [ -d "superlu" ]; then
    git clone https://github.com/xiaoyeli/superlu.git
fi
cd superlu

# run cmake
cmake . \
    -D CMAKE_BUILD_TYPE=Release \
    -D BUILD_SHARED_LIBS=ON \
    -D TPL_BLAS_LIBRARIES=openblas \
    -D TPL_ENABLE_METISLIB=ON \
    -D TPL_METIS_INCLUDE_DIRS="/usr/include" \
    -D TPL_METIS_LIBRARIES="metis"

make && make test

# copy include files
sudo mkdir -p $INCDIR/
sudo cp -av SRC/*.h $INCDIR/

# copy libray files
sudo mkdir -p $LIBDIR/
sudo cp -av SRC/*.so* /usr/local/lib/superlu/

# update ldconfig
echo "${LIBDIR}" | sudo tee /etc/ld.so.conf.d/superlu.conf >/dev/null
sudo ldconfig
