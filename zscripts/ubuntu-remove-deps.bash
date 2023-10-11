#!/bin/bash

set -e

sudo apt-get remove \
    gfortran \
    liblapacke-dev \
    libmumps-seq-dev \
    libopenblas-dev \
    libsuitesparse-dev

