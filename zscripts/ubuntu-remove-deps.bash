#!/bin/bash

set -e

sudo apt-get remove \
    gfortran \
    liblapacke-dev \
    libmetis-dev \
    libopenblas-dev \
    libsuitesparse-dev
