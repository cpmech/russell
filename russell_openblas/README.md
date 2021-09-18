# Russell OpenBLAS - Thin wrapper to some OpenBLAS routines

_This crate is part of [Russell - Rust Scientific Library](https://github.com/cpmech/russell)_

This package implements a thin wrapper to a few of the OpenBLAS routines for performing linear algebra computations.

Documentation:

- [API reference (docs.rs)](https://docs.rs/russell_openblas)

## Installation

Install some libraries:

```bash
sudo apt-get install \
    libopenblas-dev \
    liblapacke-dev
```

Add this to your Cargo.toml (choose the right version):

```toml
[dependencies]
russell_openblas = "*"
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
