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
    liblapacke-dev \
    libmumps-seq-dev \
    libopenblas-dev \
    libsuitesparse-dev
```

Add this to your Cargo.toml (replace the right version):

```toml
[dependencies]
russell_sparse = "*"
```

### Optional: Use a locally compiled MUMPS library

The standard Debian `libmumps-seq-dev` does not come with Metis or OpenMP that may lead to faster calculations. Therefore, it may be advantageous to use a locally compiled MUMPS library.

We just need the include files in `/usr/local/include/mumps` and a library file named `libdmumps_open_seq_omp` in `/usr/local/lib/mumps`.

Follow the instructions from https://github.com/cpmech/script-install-mumps and then set the environment variable `USE_LOCAL_MUMPS=1`:

```
export USE_LOCAL_MUMPS=1
```
