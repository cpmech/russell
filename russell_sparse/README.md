# Russell Sparse - Sparse matrix tools and solvers

_This crate is part of [Russell - Rust Scientific Library](https://github.com/cpmech/russell)_

This repository contains tools for handling sparse matrices and functions to solve large sparse systems using the best libraries out there, such as [UMFPACK](https://github.com/DrTimothyAldenDavis/SuiteSparse) and [MUMPS](https://mumps-solver.org).

Documentation:

- [API reference (docs.rs)](https://docs.rs/russell_sparse)

## Installation (Ubuntu/Linux)

First, you need to install some dependencies:

```bash
sudo apt install liblapacke-dev libopenblas-dev
```

Next, there are some options to compile the code with UMFPACK and MUMPS:

**Option 1.** Using the standard Debian packages. The code works just fine with the standard Debian packages. However, they may be outdated. Furthermore, the sequential version of MUMPS in Debian may be slow because it does not include Metis and OpenMP.

**Option 2.** Alternatively, the code can be linked with locally compiled MUMPS and UMFPACK. In this case, the following environment variables must be set:

```bash
export RUSSELL_SPARSE_USE_LOCAL_MUMPS=1
export RUSSELL_SPARSE_USE_LOCAL_UMFPACK=1
```

You can combine a Debian-packaged UMFPACK with a locally compiled MUMPS (or vice-versa). Just set the above variables as appropriate.

### Option 1 - Standard Debian packages

Install the following libraries:

```bash
sudo apt install libmumps-seq-dev libsuitesparse-dev
```

### Option 2 - Locally compiled MUMPS and UMFPACK

**Important:** To use the locally compiled UMFPACK, in addition to setting the above environment variable, you must **remove** `libsuitesparse-dev`.

First, install some dependencies:

```bash
bash zscripts/install-deps.bash
```

Second (optional), to download and compile MUMPS and install it in `/usr/local/include/mumps` and `/usr/local/lib/mumps`, run:

```bash
bash zscripts/install-mumps.bash
```

Note that the locally compiled MUMPS can co-exist with `libmumps-seq-dev`.

Third (optional), to download and compile UMFPACK and install it in `/usr/local/include/umfpack` and `/usr/local/lib/umfpack`, run:

```bash
bash zscripts/install-umfpack.bash
```

### Crates.io

[![Crates.io](https://img.shields.io/crates/v/russell_sparse.svg)](https://crates.io/crates/russell_sparse)

👆 Check the crate version and update your Cargo.toml accordingly:

```toml
[dependencies]
russell_sparse = "*"
```

### Number of threads

By default, OpenBLAS will use all available threads, including Hyper-Threads that may worsen the performance. Thus, it is best to set the following environment variable:

```bash
export OPENBLAS_NUM_THREADS=<real-core-number>
```

Substitute `<real-core-number>` with the correct value from your system.

Furthermore, if working on a multi-threaded application where the solver should not be multi-threaded on its own (e.g., running parallel calculations such as in optimization via genetic algorithms or differential evolution), you may set:

```bash
export OPENBLAS_NUM_THREADS=1
```

## Examples

### Solve a tiny sparse linear system using UMFPACK

```rust
use russell_chk::vec_approx_eq;
use russell_lab::{Matrix, Vector};
use russell_sparse::prelude::*;
use russell_sparse::StrError;

fn main() -> Result<(), StrError> {
    // allocate a square matrix
    let nrow = 3; // number of equations
    let ncol = nrow; // number of equations
    let nnz = 5; // number of non-zeros
    let mut coo = CooMatrix::new(None, nrow, ncol, nnz)?;
    coo.put(0, 0, 0.2)?;
    coo.put(0, 1, 0.2)?;
    coo.put(1, 0, 0.5)?;
    coo.put(1, 1, -0.25)?;
    coo.put(2, 2, 0.25)?;

    // print matrix
    let mut a = Matrix::new(nrow, ncol);
    coo.to_matrix(&mut a)?;
    let correct = "┌                   ┐\n\
                   │   0.2   0.2     0 │\n\
                   │   0.5 -0.25     0 │\n\
                   │     0     0  0.25 │\n\
                   └                   ┘";
    assert_eq!(format!("{}", a), correct);

    // allocate solver, initialize, and factorize
    let mut solver = SolverUMFPACK::new()?;
    solver.initialize(&coo)?;
    solver.factorize(&coo, false)?;

    // allocate rhs
    let rhs1 = Vector::from(&[1.0, 1.0, 1.0]);
    let rhs2 = Vector::from(&[2.0, 2.0, 2.0]);

    // calculate solution
    let mut x1 = Vector::new(nrow);
    solver.solve(&mut x1, &rhs1, false)?;
    let correct = vec![3.0, 2.0, 4.0];
    vec_approx_eq(x1.as_data(), &correct, 1e-14);

    // solve again
    let mut x2 = Vector::new(nrow);
    solver.solve(&mut x2, &rhs2, false)?;
    let correct = vec![6.0, 4.0, 8.0];
    vec_approx_eq(x2.as_data(), &correct, 1e-14);
    Ok(())
}
```

## Sparse solvers

`russell_sparse` wraps two direct sparse solvers: UMFPACK and MUMPS. The default solver is UMFPACK; however, UMFPACK may run out of memory for large matrices, whereas MUMPS still may work. On the other hand, MUMPS is **not** thread-safe and thus must be used in single-threaded applications.

## Tools

This crate includes a tool named `solve_matrix_market` to study the performance of the available sparse solvers (currently MUMPS and UMFPACK).

`solve_matrix_market` reads a [Matrix Market file](https://math.nist.gov/MatrixMarket/formats.html) and solves the linear system:

```text
A ⋅ x = rhs
```

where the right-hand side is a vector containing only ones.

The data directory contains an example of a Matrix Market file named `bfwb62.mtx`, and you may download more matrices from https://sparse.tamu.edu/

For example, run the command:

```bash
cargo run --release --bin solve_matrix_market -- ~/Downloads/matrix-market/bfwb62.mtx
```

Or

```bash
cargo run --release --bin solve_matrix_market -- --help
```

to see the options.
