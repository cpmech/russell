# Russell - Rust Scientific Library

[![codecov](https://codecov.io/gh/cpmech/russell/graph/badge.svg?token=PQWSKMZQXT)](https://codecov.io/gh/cpmech/russell)

![Bertrand Russell](Bertrand_Russell_1957.jpg)

([CC0](http://creativecommons.org/publicdomain/zero/1.0/deed.en). Photo: [Bertrand Russell](https://en.wikipedia.org/wiki/Bertrand_Russell))

## Contents

* [Introduction](#introduction)
* [Crates](#crates)
* [Installation on Debian/Ubuntu/Linux](#installation)
* [Installation on macOS](#macos)
* [Examples](#examples)
* [Todo list](#todo)
* [Code coverage](#coverage)

## <a name="introduction"></a> Introduction

**Russell** (Rust Scientific Library) assists in developing scientific computations using the Rust language. Our initial focus is on numerical methods and solvers for sparse linear systems and differential equations; however, anything is possible 😉.

The "main" crate here is [russell_lab](https://github.com/cpmech/russell/tree/main/russell_lab), a **mat**rix-vector **lab**oratory, which provides the fundamental `Vector` and `Matrix` structures and several functions to perform linear algebra computations. Thus, we recommend looking at [russell_lab](https://github.com/cpmech/russell/tree/main/russell_lab) first.

The next interesting crate is [russell_sparse](https://github.com/cpmech/russell/tree/main/russell_sparse), which implements sparse matrix structures such as COO (coordinates), CSC (compressed sparse column), and CSR (compressed sparse row) formats. `russell_sparse` also wraps powerful linear system solvers such as [UMFPACK](https://github.com/DrTimothyAldenDavis/SuiteSparse) and [MUMPS](https://mumps-solver.org).

This library aims to wrap the best solutions (e.g., UMFPACK) while maintaining a very **clean** and idiomatic Rust code. See our [Todo list](#todo). The code must also be simple to use and thoroughly tested with a minimum [coverage](#coverage) of 95%.

## <a name="crates"></a> Crates

Available crates:

- [![Crates.io](https://img.shields.io/crates/v/russell_lab.svg)](https://crates.io/crates/russell_lab) [russell_lab](https://github.com/cpmech/russell/tree/main/russell_lab) Matrix-vector laboratory including linear algebra tools (with OpenBLAS or Intel MKL)
- [![Crates.io](https://img.shields.io/crates/v/russell_sparse.svg)](https://crates.io/crates/russell_sparse) [russell_sparse](https://github.com/cpmech/russell/tree/main/russell_sparse) Sparse matrix tools and solvers (with MUMPS, UMFPACK, and Intel DSS)
- [![Crates.io](https://img.shields.io/crates/v/russell_stat.svg)](https://crates.io/crates/russell_stat) [russell_stat](https://github.com/cpmech/russell/tree/main/russell_stat) Statistics calculations, probability distributions, and pseudo random numbers
- [![Crates.io](https://img.shields.io/crates/v/russell_tensor.svg)](https://crates.io/crates/russell_tensor) [russell_tensor](https://github.com/cpmech/russell/tree/main/russell_tensor) Tensor analysis structures and functions for continuum mechanics

External associated and recommended crates:

- [plotpy](https://github.com/cpmech/plotpy) Plotting tools using Python3/Matplotlib as an engine (for quality graphics)
- [tritet](https://github.com/cpmech/tritet) Triangle and tetrahedron mesh generators (with Triangle and Tetgen)
- [gemlab](https://github.com/cpmech/gemlab) Geometry, meshes, and numerical integration for finite element analyses

## <a name="installation"></a> Installation on Debian/Ubuntu/Linux

**Russell** depends on external (non-Rust) packages for linear algebra and the solution of large sparse linear systems. For instance, the following need to be installed:

* [OpenBLAS](https://github.com/OpenMathLib/OpenBLAS) **xor** [Intel MKL](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2023-2/overview.html)
* [UMFPACK](https://github.com/DrTimothyAldenDavis/SuiteSparse) and [MUMPS](https://mumps-solver.org)
* (optional) [Intel DSS](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2023-2/direct-sparse-solver-dss-interface-routines.html)

The above packages may be installed via `apt`; however the Debian packages may lack features that boost performance (e.g., Metis ordering for MUMPS is missing in Debian). Therefore, we may compile UMFPACK and MUMPS locally and install them in `/usr/local`.

In summary, we have three options:

1. Use the standard Debian packages based on OpenBLAS (default)
2. Compile MUMPS and UMFPACK with OpenBLAS
3. Compile MUMPS and UMFPACK with Intel MKL (and enable the Intel DSS solver)

Options 2 and 3 require the following environment variables:

```bash
export RUSSELL_SPARSE_USE_LOCAL_MUMPS=1
export RUSSELL_SPARSE_USE_LOCAL_UMFPACK=1
```

Option 3 also requires the following environment variables:

```bash
export RUSSELL_LAB_USE_INTEL_MKL=1
export RUSSELL_SPARSE_WITH_INTEL_DSS=1
```

For convenience, you may use the scripts in the [russell/zscripts](https://github.com/cpmech/russell/tree/main/zscripts) directory.

**1.** Use the standard Debian packages based on OpenBLAS:

```bash
bash zscripts/01-ubuntu-openblas-debian.bash
```

**2. (xor)** compile MUMPS and UMFPACK with OpenBLAS:

```bash
bash zscripts/02-ubuntu-openblas-compile.bash
```

**3. (xor)** compile MUMPS and UMFPACK with Intel MKL:

```bash
bash zscripts/03-ubuntu-intel-mkl-compile.bash
```

The compiled MUMPS files will be installed in `/usr/local/include/mumps` and `/usr/local/lib/mumps`.

The compiled UMFPACK files will be installed in `/usr/local/include/umfpack` and `/usr/local/lib/umfpack`.

### Number of threads

By default, OpenBLAS will use all available threads, including Hyper-Threads that may worsen the performance. Thus, it is best to set the following environment variable:

```bash
export OPENBLAS_NUM_THREADS=<real-core-number>
```

Substitute `<real-core-number>` with the correct value from your system.

Furthermore, if working on a multi-threaded application where the solver should not be multi-threaded on its own (e.g., running parallel calculations in an optimization tool), you may set:

```bash
export OPENBLAS_NUM_THREADS=1
```

## <a name="macos"></a> Installation on macOS

At this time, only OpenBLAS has been tested on macOS---we still need to study how to use MUMPS and UMFPACK on macOS.

First, install [Homebrew](https://brew.sh/). Then, run:

```bash
brew install lapack openblas
```

Next, we must set the `LIBRARY_PATH`:

```bash
export LIBRARY_PATH=$LIBRARY_PATH:$(brew --prefix)/opt/lapack/lib:$(brew --prefix)/opt/openblas/lib
```

## <a name="examples"></a> Examples

See also:

* [russell_lab/examples](https://github.com/cpmech/russell/tree/main/russell_lab/examples)
* [russell_sparse/examples](https://github.com/cpmech/russell/tree/main/russell_sparse/examples)

### Compute a singular value decomposition

```rust
use russell_lab::{mat_svd, Matrix, Vector, StrError};

fn main() -> Result<(), StrError> {
    // set matrix
    let mut a = Matrix::from(&[
        [2.0, 4.0],
        [1.0, 3.0],
        [0.0, 0.0],
        [0.0, 0.0],
    ]);

    // allocate output structures
    let (m, n) = a.dims();
    let min_mn = if m < n { m } else { n };
    let mut s = Vector::new(min_mn);
    let mut u = Matrix::new(m, m);
    let mut vt = Matrix::new(n, n);

    // perform SVD
    mat_svd(&mut s, &mut u, &mut vt, &mut a)?;

    // check S
    let s_correct = "┌      ┐\n\
                     │ 5.46 │\n\
                     │ 0.37 │\n\
                     └      ┘";
    assert_eq!(format!("{:.2}", s), s_correct);

    // check SVD: a == u * s * vt
    let mut usv = Matrix::new(m, n);
    for i in 0..m {
        for j in 0..n {
            for k in 0..min_mn {
                usv.add(i, j, u.get(i, k) * s[k] * vt.get(k, j));
            }
        }
    }
    let usv_correct = "┌                   ┐\n\
                       │ 2.000000 4.000000 │\n\
                       │ 1.000000 3.000000 │\n\
                       │ 0.000000 0.000000 │\n\
                       │ 0.000000 0.000000 │\n\
                       └                   ┘";
    assert_eq!(format!("{:.6}", usv), usv_correct);
    Ok(())
}
```

### Cholesky factorization

```rust
use russell_lab::*;

fn main() -> Result<(), StrError> {
    // set matrix (full)
    #[rustfmt::skip]
    let a_full = Matrix::from(&[
        [ 3.0, 0.0,-3.0, 0.0],
        [ 0.0, 3.0, 1.0, 2.0],
        [-3.0, 1.0, 4.0, 1.0],
        [ 0.0, 2.0, 1.0, 3.0],
    ]);

    // set matrix (lower)
    #[rustfmt::skip]
    let mut a_lower = Matrix::from(&[
        [ 3.0, 0.0, 0.0, 0.0],
        [ 0.0, 3.0, 0.0, 0.0],
        [-3.0, 1.0, 4.0, 0.0],
        [ 0.0, 2.0, 1.0, 3.0],
    ]);

    // set matrix (upper)
    #[rustfmt::skip]
    let mut a_upper = Matrix::from(&[
        [3.0, 0.0,-3.0, 0.0],
        [0.0, 3.0, 1.0, 2.0],
        [0.0, 0.0, 4.0, 1.0],
        [0.0, 0.0, 0.0, 3.0],
    ]);

    // perform Cholesky factorization (lower)
    mat_cholesky(&mut a_lower, false)?;
    let l = &a_lower;

    // perform Cholesky factorization (upper)
    mat_cholesky(&mut a_upper, true)?;
    let u = &a_upper;

    // check:  l ⋅ lᵀ = a
    let m = a_full.nrow();
    let mut l_lt = Matrix::new(m, m);
    for i in 0..m {
        for j in 0..m {
            for k in 0..m {
                l_lt.add(i, j, l.get(i, k) * l.get(j, k));
            }
        }
    }
    mat_approx_eq(&l_lt, &a_full, 1e-14);

    // check:   uᵀ ⋅ u = a
    let mut ut_u = Matrix::new(m, m);
    for i in 0..m {
        for j in 0..m {
            for k in 0..m {
                ut_u.add(i, j, u.get(k, i) * u.get(k, j));
            }
        }
    }
    mat_approx_eq(&ut_u, &a_full, 1e-14);
    Ok(())
}
```

### Solve a tiny (dense) linear system

```rust
use russell_lab::{solve_lin_sys, Matrix, Vector, StrError};

fn main() -> Result<(), StrError> {
    // set matrix and right-hand side
    let mut a = Matrix::from(&[
        [1.0,  3.0, -2.0],
        [3.0,  5.0,  6.0],
        [2.0,  4.0,  3.0],
    ]);
    let mut b = Vector::from(&[5.0, 7.0, 8.0]);

    // solve linear system b := a⁻¹⋅b
    solve_lin_sys(&mut b, &mut a)?;

    // check
    let x_correct = "┌         ┐\n\
                     │ -15.000 │\n\
                     │   8.000 │\n\
                     │   2.000 │\n\
                     └         ┘";
    assert_eq!(format!("{:.3}", b), x_correct);
    Ok(())
}
```

### Solve a small sparse linear system using UMFPACK

```rust
use russell_lab::*;
use russell_sparse::prelude::*;
use russell_sparse::StrError;

fn main() -> Result<(), StrError> {
    // constants
    let ndim = 5; // number of rows = number of columns
    let nnz = 13; // number of non-zero values, including duplicates

    // allocate solver
    let mut umfpack = SolverUMFPACK::new()?;

    // allocate the coefficient matrix
    //  2  3  .  .  .
    //  3  .  4  .  6
    //  . -1 -3  2  .
    //  .  .  1  .  .
    //  .  4  2  .  1
    let mut coo = SparseMatrix::new_coo(ndim, ndim, nnz, None, false)?;
    coo.put(0, 0, 1.0)?; // << (0, 0, a00/2) duplicate
    coo.put(0, 0, 1.0)?; // << (0, 0, a00/2) duplicate
    coo.put(1, 0, 3.0)?;
    coo.put(0, 1, 3.0)?;
    coo.put(2, 1, -1.0)?;
    coo.put(4, 1, 4.0)?;
    coo.put(1, 2, 4.0)?;
    coo.put(2, 2, -3.0)?;
    coo.put(3, 2, 1.0)?;
    coo.put(4, 2, 2.0)?;
    coo.put(2, 3, 2.0)?;
    coo.put(1, 4, 6.0)?;
    coo.put(4, 4, 1.0)?;

    // parameters
    let mut params = LinSolParams::new();
    params.verbose = false;
    params.compute_determinant = true;

    // call factorize
    umfpack.factorize(&mut coo, Some(params))?;

    // allocate x and rhs
    let mut x = Vector::new(ndim);
    let rhs = Vector::from(&[8.0, 45.0, -3.0, 3.0, 19.0]);

    // calculate the solution
    umfpack.solve(&mut x, &coo, &rhs, false)?;
    println!("x =\n{}", x);

    // check the results
    let correct = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    vec_approx_eq(x.as_data(), &correct, 1e-14);

    // analysis
    let mut stats = StatsLinSol::new();
    umfpack.update_stats(&mut stats);
    let (mx, ex) = (stats.determinant.mantissa, stats.determinant.exponent);
    println!("det(a) = {:?}", mx * f64::powf(10.0, ex));
    println!("rcond  = {:?}", stats.output.umfpack_rcond_estimate);
    Ok(())
}
```

## <a name="todo"></a> Todo list

- [ ] Improve crate `russell_lab`
    - [x] Implement more integration tests for linear algebra
    - [x] Implement more examples
    - [ ] Implement more benchmarks
    - [x] Wrap Intel MKL (option for OpenBLAS)
    - [x] Add more complex numbers functions
    - [ ] Add fundamental functions to `russell_lab`
        - [ ] Implement the modified Bessel functions
    - [ ] Implement some numerical methods in `russell_lab`
        - [ ] Implement Brent's solver
        - [ ] Implement solver for the cubic equation
        - [x] Implement numerical derivation
        - [x] Implement numerical Jacobian function
        - [ ] Implement Newton's method for nonlinear systems
        - [ ] Implement numerical quadrature
    - [ ] Add interpolation and polynomials to `russell_lab`
        - [ ] Implement Chebyshev interpolation and polynomials
        - [ ] Implement Orthogonal polynomials
        - [ ] Implement Lagrange interpolation
    - [ ] Implement FFT
- [x] Improve the `russell_sparse` crate
    - [x] Implement the Compressed Sparse Column format (CSC)
    - [x] Implement the Compressed Sparse Row format (CSC)
    - [x] Improve the C-interface to UMFPACK and MUMPS
    - [x] Implement the C-interface to Intel DSS
    - [ ] Write the conversion from COO to CSC in Rust
    - [ ] Possibly re-write (after benchmarking) the conversion from COO to CSR
    - [ ] Re-study the possibility to wrap SuperLU (see deleted branch)
- [ ] Improve crate `russell_stat`
    - [x] Add probability distribution functions
    - [x] Implement drawing of ASCII histograms
- [ ] Improve the `russell_tensor` crate
    - [x] Implement functions to calculate invariants
    - [x] Implement first and second order derivatives of invariants
    - [x] Implement some high-order derivatives
    - [ ] Implement standard continuum mechanics tensors
- [ ] Make it possible to install Russell on Windows and macOS 
    - [ ] Use Intel MKL on Windows
    - [ ] Install MUMPS and UMFPACK on Windows and macOS

## <a name="coverage"></a> Code coverage

### Sunburst

The inner-most circle is the entire project, moving away from the center are folders then, finally, a single file. The size and color of each slice is representing the number of statements and the coverage, respectively.

![Sunburst](https://codecov.io/gh/cpmech/russell/graphs/sunburst.svg?token=PQWSKMZQXT)

### Grid

Each block represents a single file in the project. The size and color of each block is represented by the number of statements and the coverage, respectively.

![Grid](https://codecov.io/gh/cpmech/russell/graphs/tree.svg?token=PQWSKMZQXT)

### Icicle

The top section represents the entire project. Proceeding with folders and finally individual files. The size and color of each slice is representing the number of statements and the coverage, respectively.

![Icicle](https://codecov.io/gh/cpmech/russell/graphs/icicle.svg?token=PQWSKMZQXT)
