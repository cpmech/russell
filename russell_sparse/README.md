# Russell Sparse - Sparse matrix tools and solvers

_This crate is part of [Russell - Rust Scientific Library](https://github.com/cpmech/russell)_

Work in progress...

This repository contains tools for handling sparse matrices and functions to solve large sparse systems.

Documentation:

- [API reference (docs.rs)](https://docs.rs/russell_sparse)

## Installation

Install the following Debian packages:

```bash
sudo apt-get install \
    libopenblas-dev \
    liblapacke-dev \
    libsuitesparse-dev
```

Compile and install the MUMPS solver using the procedure explained in https://github.com/cpmech/script-install-mumps

Add this to your Cargo.toml (replace the right version):

```toml
[dependencies]
russell_sparse = "*"
```
