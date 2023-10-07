# Russell Sparse - Sparse matrix tools and solvers

_This crate is part of [Russell - Rust Scientific Library](https://github.com/cpmech/russell)_

This repository implements the `russell_sparse` crate which contains tools for handling sparse matrices and functions to solve large sparse systems using the best libraries out there, such as [UMFPACK (recommended)](https://github.com/DrTimothyAldenDavis/SuiteSparse) and [MUMPS (for very large systems)](https://mumps-solver.org). Optionally, you may want to use the [Intel DSS solver](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2023-2/direct-sparse-solver-dss-interface-routines.html).

We have three storage formats for sparse matrices:

* COO: COOrdinates matrix, also known as a sparse triplet.
* CSC: Compressed Sparse Column matrix
* CSR: Compressed Sparse Row matrix

Additionally, to unify the handling of the above sparse matrix data structures, we have:

* SparseMatrix: Either a COO, CSC, or CSR matrix

The COO matrix is the best when we need to update the values of the matrix because it has easy access to the triples (i, j, aij). For instance, the repetitive access is the primary use case for codes based on the finite element method (FEM) for approximating partial differential equations. Moreover, the COO matrix allows storing duplicate entries; for example, the triple `(0, 0, 123.0)` can be stored as two triples `(0, 0, 100.0)` and `(0, 0, 23.0)`. Again, this is the primary need for FEM codes because of the so-called assembly process where elements add to the same positions in the "global stiffness" matrix. Nonetheless, the duplicate entries must be summed up at some stage for the linear solver (e.g., MUMPS, UMFPACK, and Intel DSS). These linear solvers also use the more memory-efficient storage formats CSC and CSR. See the [russell_sparse documentation](https://docs.rs/russell_sparse) for further information.

This library also provides functions to read and write Matrix Market files containing (huge) sparse matrices that can be used in performance benchmarking or other studies. The [read_matrix_market()] function reads a Matrix Market file and returns a [CooMatrix]. To write a Matrix Market file, we can use the function [write_matrix_market()], which takes a [SparseMatrix] and, thus, automatically convert COO to CSC or COO to CSR, also performing the sum of duplicates. The `write_matrix_market` also writes an SMAT file (almost like the Matrix Market format) without the header and with zero-based indices. The SMAT file can be given to the fantastic [Vismatrix](https://github.com/cpmech/vismatrix) tool to visualize the sparse matrix structure and values interactively; see the example below.

![readme-vismatrix](https://raw.githubusercontent.com/cpmech/russell/main/russell_sparse/data/figures/readme-vismatrix.png)

See the documentation for further information:

- [russell_sparse documentation](https://docs.rs/russell_sparse) - Contains the API reference and examples

## Installation (Ubuntu/Linux)

### Required libraries

First, you need to install some dependencies:

```bash
sudo apt install liblapacke-dev libopenblas-dev
```

Next, there are some options to compile the code with [UMFPACK](https://github.com/DrTimothyAldenDavis/SuiteSparse) and [MUMPS](https://mumps-solver.org):

**Option 1.** Using the standard Debian packages. The code works just fine with the standard Debian packages. However, they may be outdated. Furthermore, the sequential version of MUMPS in Debian may be slow because it does not include Metis and OpenMP.

**Option 2.** Alternatively, the code can be linked with locally compiled MUMPS and UMFPACK. In this case, the following environment variables must be set:

```bash
export RUSSELL_SPARSE_USE_LOCAL_MUMPS=1
export RUSSELL_SPARSE_USE_LOCAL_UMFPACK=1
```

You can combine a Debian packages libraries with locally compiled ones; just set the above variables as appropriate.

#### Option 1 - Standard Debian packages

Install the following libraries:

```bash
sudo apt install libmumps-seq-dev libsuitesparse-dev
```

#### Option 2 - Locally compiled MUMPS and UMFPACK

**Important:** To use the locally compiled UMFPACK, in addition to setting the above environment variable, you must **remove** `libsuitesparse-dev`.

First, install some dependencies:

```bash
bash zscripts/install-deps.bash
```

Second, to download and compile MUMPS and install it in `/usr/local/include/mumps` and `/usr/local/lib/mumps`, run:

```bash
bash zscripts/install-mumps.bash
```

Note that the locally compiled MUMPS can co-exist with `libmumps-seq-dev`.

Third, to download and compile UMFPACK and install it in `/usr/local/include/umfpack` and `/usr/local/lib/umfpack`, run:

```bash
bash zscripts/install-umfpack.bash
```

### (optional) Intel DSS solver

If you wish to use the [Intel DSS](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2023-2/direct-sparse-solver-dss-interface-routines.html) solver, you need the additional Debian packages: `intel-oneapi-mkl` and `intel-oneapi-mkl-devel`, that can be easily installed by running:

```bash
bash zscripts/install-intel-mkl-linux.bash
```

Furthermore, you need to define the following environment variable:

```bash
export RUSSELL_SPARSE_WITH_INTEL_DSS=1
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
use russell_lab::{vec_approx_eq, Vector};
use russell_sparse::prelude::*;
use russell_sparse::StrError;

fn main() -> Result<(), StrError> {
    // constants
    let ndim = 3; // number of rows = number of columns
    let nnz = 5; // number of non-zero values

    // allocate solver
    let mut umfpack = SolverUMFPACK::new()?;

    // allocate the coefficient matrix
    let mut coo = SparseMatrix::new_coo(ndim, ndim, nnz, None, false)?;
    coo.put(0, 0, 0.2)?;
    coo.put(0, 1, 0.2)?;
    coo.put(1, 0, 0.5)?;
    coo.put(1, 1, -0.25)?;
    coo.put(2, 2, 0.25)?;

    // print matrix
    let a = coo.as_dense();
    let correct = "┌                   ┐\n\
                   │   0.2   0.2     0 │\n\
                   │   0.5 -0.25     0 │\n\
                   │     0     0  0.25 │\n\
                   └                   ┘";
    assert_eq!(format!("{}", a), correct);

    // call factorize
    umfpack.factorize(&mut coo, None)?;

    // allocate two right-hand side vectors
    let b = Vector::from(&[1.0, 1.0, 1.0]);

    // calculate the solution
    let mut x = Vector::new(ndim);
    umfpack.solve(&mut x, &coo, &b, false)?;
    let correct = vec![3.0, 2.0, 4.0];
    vec_approx_eq(x.as_data(), &correct, 1e-14);
    Ok(())
}
```

See [russell_sparse documentation](https://docs.rs/russell_sparse) for more examples.

See also the folder `examples`.

## Tools

This crate includes a tool named `solve_matrix_market` to study the performance of the available sparse solvers (currently MUMPS and UMFPACK).

`solve_matrix_market` reads a [Matrix Market file](https://math.nist.gov/MatrixMarket/formats.html) and solves the linear system:

```text
A ⋅ x = b
```

where the right-hand side (b) is a vector containing only ones.

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

The default solver of `solve_matrix_market` is UMFPACK. To run with MUMPS, use the `--genie` (-g) flag:

```bash
cargo run --release --bin solve_matrix_market -- -g mumps -x -y ~/Downloads/matrix-market/bfwb62.mtx
```

The output looks like this:

```json
{
  "main": {
    "platform": "Russell",
    "blas_lib": "OpenBLAS",
    "solver": "MUMPS-local"
  },
  "matrix": {
    "name": "bfwb62",
    "nrow": 62,
    "ncol": 62,
    "nnz": 202,
    "symmetry": "Some(General(Lower))"
  },
  "requests": {
    "ordering": "Auto",
    "scaling": "Auto",
    "mumps_openmp_num_threads": 1
  },
  "output": {
    "effective_ordering": "Amf",
    "effective_scaling": "RowColIter",
    "openmp_num_threads": 32,
    "umfpack_strategy": "Unknown",
    "umfpack_rcond_estimate": 0.0
  },
  "determinant": {
    "mantissa": 0.0,
    "base": 2.0,
    "exponent": 0.0
  },
  "verify": {
    "max_abs_a": 0.0001,
    "max_abs_ax": 1.0000000000000004,
    "max_abs_diff": 5.551115123125783e-16,
    "relative_error": 5.550560067119071e-16
  },
  "time_human": {
    "read_matrix": "37.654µs",
    "factorize": "5.117697ms",
    "solve": "219.659µs",
    "total_f_and_s": "5.337356ms",
    "verify": "6.638µs"
  },
  "time_nanoseconds": {
    "read_matrix": 37654,
    "factorize": 5117697,
    "solve": 219659,
    "total_f_and_s": 5337356,
    "verify": 6638
  },
  "mumps_stats": {
    "inf_norm_a": 0.00021250000000000002,
    "inf_norm_x": 116611.5333525506,
    "scaled_residual": 2.1281557019905676e-17,
    "backward_error_omega1": 2.1420239141348292e-16,
    "backward_error_omega2": 0.0,
    "normalized_delta_x": 7.516512844561073e-16,
    "condition_number1": 3.50907046133377,
    "condition_number2": 1.0
  }
}
```
