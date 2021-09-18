# Russell Lab - Matrix-vector laboratory including linear algebra tools

_This crate is part of [Russell - Rust Scientific Library](https://github.com/cpmech/russell)_

This repository is a "rust laboratory" for vectors and matrices.

Documentation:

- [API reference (docs.rs)](https://docs.rs/russell_lab)

## Installation

Install some libraries:

```bash
sudo apt-get install \
    liblapacke-dev \
    libopenblas-dev
```

Add this to your Cargo.toml (choose the right version):

```toml
[dependencies]
russell_lab = "*"
```

### Number of threads

By default OpenBLAS will use all available threads, including Hyper-Threads that make the performance worse. Thus, it is best to set the following environment variable:

```
export OPENBLAS_NUM_THREADS=<real-core-count>
```

Furthermore, if working on a multi-threaded application, it is recommended to set:

```
export OPENBLAS_NUM_THREADS=1
```
