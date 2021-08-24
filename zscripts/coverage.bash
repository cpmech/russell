#!/bin/bash

# install:
#   rustup component add llvm-tools-preview --toolchain nightly
#   cargo install cargo-llvm-cov

# reference:
#   https://github.com/taiki-e/cargo-llvm-cov

cargo llvm-cov --open
