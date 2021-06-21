#!/bin/bash

# install:
#   rustup component add llvm-tools-preview --toolchain nightly
#   cargo install cargo-llvm-cov --version 0.1.0-alpha.4

# reference:
#   https://github.com/taiki-e/cargo-llvm-cov

cargo llvm-cov --open
