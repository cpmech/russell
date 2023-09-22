# Russell Sparse - Sparse matrix tools and solvers

_This crate is part of [Russell - Rust Scientific Library](https://github.com/cpmech/russell)_

This repository contains tools for handling sparse matrices and functions to solve large sparse systems.

Documentation:

- [API reference (docs.rs)](https://docs.rs/russell_sparse)

## Installation

Essential dependencies:

```bash
sudo apt-get install liblapacke-dev libopenblas-dev libsuitesparse-dev
```

**Important:** The Debian `libmumps-seq-dev` package does not come with Metis or OpenMP, which makes it possible slower. Therefore, it may be advantageous to use a locally compiled MUMPS library with Metis and OpenMP. Below we recommend Option 1, but Option 2 is also available.

### Option 1: Locally compiled MUMPS solver

Follow the steps in https://github.com/cpmech/script-install-mumps and set the environment variable:

```bash
export USE_LOCAL_MUMPS=1
```

### Option 2: Debian/Ubuntu package for the MUMPS solver

Install:

```shell
sudo apt-get install libmumps-seq-dev
```

### Crates.io

[![Crates.io](https://img.shields.io/crates/v/russell_sparse.svg)](https://crates.io/crates/russell_sparse)

👆 Check the crate version and update your Cargo.toml accordingly:

```toml
[dependencies]
russell_sparse = "*"
```

### Number of threads

By default OpenBLAS will use all available threads, including Hyper-Threads which may make the performance worse. Thus, it is best to set the following environment variable:

```bash
export OPENBLAS_NUM_THREADS=<real-core-count>
```

Furthermore, if working on a multi-threaded application where the solver should not be multi-threaded, you may set:

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

`russell_sparse` wraps two direct sparse solvers, namely, UMFPACK and MUMPS. The default solver is UMFPACK; however UMFPACK may run out of memory for large matrices, whereas MUMPS still may work. On the other hand, the MUMPS solver is **not** thread-safe and thus must be used in single-threaded applications.

## Tools

This crate includes a tool named `solve_matrix_market_build` to study the performance of the available sparse solvers (currently MUMPS and UMFPACK). The `_build` suffix is to disable the coverage tool.

`solve_matrix_market_build` reads a [Matrix Market file](https://math.nist.gov/MatrixMarket/formats.html) and solves the linear system:

```text
a ⋅ x = rhs
```

with a right-hand-side containing only ones.

The data directory contains an example of Matrix Market file named `bfwb62.mtx` and you may download more matrices from https://sparse.tamu.edu/

Run the command:

```bash
cargo run --release --bin solve_matrix_market_build -- ~/Downloads/matrix-market/bfwb62.mtx
```

Or

```bash
cargo run --release --bin solve_matrix_market_build -- --help
```

for more options.
