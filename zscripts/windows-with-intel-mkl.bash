#!/bin/bash

set -e

pip install --upgrade pip
python3 -m pip debug --verbose
# pip install mkl
#pip install intel-fortran-rt
#pip install mkl-devel
