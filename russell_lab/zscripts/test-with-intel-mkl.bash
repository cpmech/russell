#!/bin/bash

NAME=mat_gen_eigen_works

cargo test $NAME --features intel_mkl -- --nocapture
