# Russell Sparse - Sparse matrix tools and solvers

_This crate is part of [Russell - Rust Scientific Library](https://github.com/cpmech/russell)_

This repository contains tools for handling sparse matrices and functions to solve large sparse systems.

Documentation:

- [API reference (docs.rs)](https://docs.rs/russell_sparse)

## Installation

Install some libraries:

```bash
sudo apt-get install \
    liblapacke-dev \
    libmumps-seq-dev \
    libopenblas-dev \
    libsuitesparse-dev
```

[![Crates.io](https://img.shields.io/crates/v/russell_sparse.svg)](https://crates.io/crates/russell_sparse)

👆 Check the crate version and update your Cargo.toml accordingly:

```toml
[dependencies]
russell_sparse = "*"
```

### Optional: Use a locally compiled MUMPS library

The standard Debian `libmumps-seq-dev` does not come with Metis or OpenMP that may lead to faster calculations. Therefore, it may be advantageous to use a locally compiled MUMPS library.

We just need the include files in `/usr/local/include/mumps` and a library file named `libdmumps_open_seq_omp` in `/usr/local/lib/mumps`.

Follow the instructions from https://github.com/cpmech/script-install-mumps and then set the environment variable `USE_LOCAL_MUMPS=1`:

```bash
export USE_LOCAL_MUMPS=1
```

### Number of threads

By default OpenBLAS will use all available threads, including Hyper-Threads that make the performance worse. Thus, it is best to set the following environment variable:

```bash
export OPENBLAS_NUM_THREADS=<real-core-count>
```

Furthermore, if working on a multi-threaded application, it is recommended to set:

```bash
export OPENBLAS_NUM_THREADS=1
```

## Examples

### Solve a sparse linear system

```rust
use russell_lab::{Matrix, Vector};
use russell_sparse::{ConfigSolver, Solver, SparseTriplet, StrError};

fn main() -> Result<(), StrError> {
    // allocate a square matrix
    let neq = 3; // number of equations
    let nnz = 5; // number of non-zeros
    let mut trip = SparseTriplet::new(neq, nnz)?;
    trip.put(0, 0, 0.2)?;
    trip.put(0, 1, 0.2)?;
    trip.put(1, 0, 0.5)?;
    trip.put(1, 1, -0.25)?;
    trip.put(2, 2, 0.25)?;
    
    // print matrix
    let mut a = Matrix::new(neq, neq);
    trip.to_matrix(&mut a)?;
    let correct = "┌                   ┐\n\
                   │   0.2   0.2     0 │\n\
                   │   0.5 -0.25     0 │\n\
                   │     0     0  0.25 │\n\
                   └                   ┘";
    assert_eq!(format!("{}", a), correct);
    
    // allocate rhs
    let rhs1 = Vector::from(&[1.0, 1.0, 1.0]);
    let rhs2 = Vector::from(&[2.0, 2.0, 2.0]);
    
    // calculate solution
    let config = ConfigSolver::new();
    let (mut solver, x1) = Solver::compute(config, &trip, &rhs1)?;
    let correct1 = "┌   ┐\n\
                    │ 3 │\n\
                    │ 2 │\n\
                    │ 4 │\n\
                    └   ┘";
    assert_eq!(format!("{}", x1), correct1);
    
    // solve again
    let mut x2 = Vector::new(neq);
    solver.solve(&mut x2, &rhs2)?;
    let correct2 = "┌   ┐\n\
                    │ 6 │\n\
                    │ 4 │\n\
                    │ 8 │\n\
                    └   ┘";
    assert_eq!(format!("{}", x2), correct2);
    Ok(())
}
```

## Sparse solvers

We wrap two direct sparse solvers: UMFPACK (aka **UMF**) and MUMPS (aka **MMP**). The default solver is UMF; however UMF may run out of memory for large matrices, whereas MMP still may work. The MMP solver is **not** thread-safe and thus must be used in single-threaded applications.

## Tools

This crate includes a tool named `solve_mm_build` to study the performance of the available sparse solvers (currently MMP and UMF). The `_build` suffix is to disable the coverage tool.

`solve_mm_build` reads a [Matrix Market file](https://math.nist.gov/MatrixMarket/formats.html) and solves the linear system:

```text
a ⋅ x = rhs
```

with a right-hand-side containing only ones.

The data directory contains an example of Matrix Market file named `bfwb62.mtx` and you may download more matrices from https://sparse.tamu.edu/

Run the command:

```bash
cargo run --release --bin solve_mm_build -- data/matrix_market/bfwb62.mtx
```

Or

```bash
cargo run --release --bin solve_mm_build -- --help
```

for more options.
