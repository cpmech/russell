# Russell Sparse - Sparse matrix tools and solvers

_This crate is part of [Russell - Rust Scientific Library](https://github.com/cpmech/russell)_

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

### Number of threads

By default OpenBLAS will use all available threads, including Hyper-Threads that make the performance worse. Thus, it is best to set the following environment variable:

```
export OPENBLAS_NUM_THREADS=<real-core-count>
```

Furthermore, if working on a multi-threaded application, it is recommended to set:

```
export OPENBLAS_NUM_THREADS=1
```

## Sparse solvers

We wrap two direct sparse solvers: UMFPACK (aka **UMF**) and MUMPS (aka **MMP**). The default solver is UMF; however UMF may run out of memory for large matrices, whereas MMP still may work. The MMP solver is **not** thread-safe and thus must be used in single-threaded applications.

## Tools

This crate includes a tool named `solve_mm_build` to study the performance of the available sparse solvers (currently MMP and UMF). The `_build` suffix is to disable the coverage tool.

`solve_mm_build` reads a [Matrix Market file](https://math.nist.gov/MatrixMarket/formats.html) and solves the linear system:

```
a ⋅ x = rhs
```

with a right-hand-side containing only ones.

The data directory contains an example of Matrix Market file named `bfwb62.mtx` and you may download more matrices from https://sparse.tamu.edu/

Run the command:

```
cargo run --release --bin solve_mm_build -- data/matrix_market/bfwb62.mtx
```

Or

```
cargo run --release --bin solve_mm_build -- --help
```

for more options.
