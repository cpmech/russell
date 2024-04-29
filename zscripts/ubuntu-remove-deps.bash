#!/bin/bash

set -e

sudo apt-get remove \
    gfortran \
    liblapacke-dev \
    libmumps-dev \
    libopenblas-dev \
    libsuitesparse-dev

