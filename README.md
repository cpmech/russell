# Russell - Rust Scientific Library

[![codecov](https://codecov.io/gh/cpmech/russell/graph/badge.svg?token=PQWSKMZQXT)](https://codecov.io/gh/cpmech/russell)
[![Test & Coverage](https://github.com/cpmech/russell/actions/workflows/test_and_coverage.yml/badge.svg)](https://github.com/cpmech/russell/actions/workflows/test_and_coverage.yml)
[![Test with local libs](https://github.com/cpmech/russell/actions/workflows/test_with_local_libs.yml/badge.svg)](https://github.com/cpmech/russell/actions/workflows/test_with_local_libs.yml)
[![Test with Intel MKL](https://github.com/cpmech/russell/actions/workflows/test_with_intel_mkl.yml/badge.svg)](https://github.com/cpmech/russell/actions/workflows/test_with_intel_mkl.yml)

![Bertrand Russell](Bertrand_Russell_1957.jpg)

([CC0](http://creativecommons.org/publicdomain/zero/1.0/deed.en). Photo: [Bertrand Russell](https://en.wikipedia.org/wiki/Bertrand_Russell))

## Contents

* [Introduction](#introduction)
* [Crates](#crates)
* [Installation on Debian/Ubuntu/Linux](#installation)
* [Installation on macOS](#macos)
* [Number of threads](#threads)
* [Examples](#examples)
* [Todo list](#todo)
* [Code coverage](#coverage)

## <a name="introduction"></a> Introduction

**Russell** (Rust Scientific Library) assists in developing scientific computations using the Rust language.

The "main" crate here is [russell_lab](https://github.com/cpmech/russell/tree/main/russell_lab), a **mat**rix-vector **lab**oratory, which provides the fundamental `Vector` and `Matrix` structures and several functions to perform linear algebra computations. Thus, we recommend looking at [russell_lab](https://github.com/cpmech/russell/tree/main/russell_lab) first.

Next, we recommend looking at the [russell_sparse](https://github.com/cpmech/russell/tree/main/russell_sparse) crate, which implements sparse matrix structures such as COO (coordinates), CSC (compressed sparse column), and CSR (compressed sparse row) formats. `russell_sparse` also wraps powerful linear system solvers such as [UMFPACK](https://github.com/DrTimothyAldenDavis/SuiteSparse) and [MUMPS](https://mumps-solver.org).

## <a name="crates"></a> Crates

Available crates:

- [![Crates.io](https://img.shields.io/crates/v/russell_lab.svg)](https://crates.io/crates/russell_lab) [russell_lab](https://github.com/cpmech/russell/tree/main/russell_lab) Matrix-vector laboratory for linear algebra (with OpenBLAS or Intel MKL)
- [![Crates.io](https://img.shields.io/crates/v/russell_sparse.svg)](https://crates.io/crates/russell_sparse) [russell_sparse](https://github.com/cpmech/russell/tree/main/russell_sparse) Sparse matrix tools and solvers (with MUMPS, UMFPACK, and Intel DSS)
- [![Crates.io](https://img.shields.io/crates/v/russell_stat.svg)](https://crates.io/crates/russell_stat) [russell_stat](https://github.com/cpmech/russell/tree/main/russell_stat) Statistics calculations, probability distributions, and pseudo random numbers
- [![Crates.io](https://img.shields.io/crates/v/russell_tensor.svg)](https://crates.io/crates/russell_tensor) [russell_tensor](https://github.com/cpmech/russell/tree/main/russell_tensor) Tensor analysis structures and functions for continuum mechanics

External associated and recommended crates:

- [plotpy](https://github.com/cpmech/plotpy) Plotting tools using Python3/Matplotlib as an engine (for quality graphics)
- [tritet](https://github.com/cpmech/tritet) Triangle and tetrahedron mesh generators (with Triangle and Tetgen)
- [gemlab](https://github.com/cpmech/gemlab) Geometry, meshes, and numerical integration for finite element analyses

## <a name="installation"></a> Installation on Debian/Ubuntu/Linux

**Russell** depends on external (non-Rust) packages for linear algebra and the solution of large sparse linear systems. The following libraries are required:

* [OpenBLAS](https://github.com/OpenMathLib/OpenBLAS) or [Intel MKL](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2023-2/overview.html)
* [MUMPS](https://mumps-solver.org) and [UMFPACK](https://github.com/DrTimothyAldenDavis/SuiteSparse)

Note that MUMPS in Debian lacks some features (e.g., Metis). Also, MUMPS in Debian is linked with OpenMPI, which may cause issues when using other MPI libraries (see, e.g., [msgpass](https://github.com/cpmech/msgpass)). Thus, an option is available to use **locally compiled** MUMPS (and UMFPACK). Furthermore, when using Intel MKL, MUMPS and UMFPACK must be locally compiled because they need to be linked with the MKL libraries.

In summary, the following options are available:

* **Case A:** OpenBLAS with the default Debian libraries
* **Case A:** OpenBLAS with locally compiled libraries
* **Case B:** Intel MKL with locally compiled libraries

### Case A: OpenBLAS

#### Default Debian packages

Run:

```bash
bash case-a-openblas-debian.bash
```

#### Locally compiled libraries (feature = local_libs)

Run:

```bash
bash case-a-openblas-local-libs.bash
```

Then, add `local_libs` to your Cargo.toml or use `cargo build --features local_libs`

### Case B: Intel MKL (feature = intel_mkl)

Run:

```bash
bash case-b-intel-mkl-local-libs.bash
```

Then, add `intel_mkl` to your Cargo.toml or use `cargo build --features intel_mkl` (note that the `local_libs` feature will be automatically enabled).

### Resulting files

If locally compiled, the above scripts will save the resulting files in `/usr/local/lib/{mumps,umfpack}` and `/usr/local/include/{mumps,umfpack}`.

## <a name="macos"></a> Installation on macOS

Currently, only OpenBLAS has been tested on macOS.

First, install [Homebrew](https://brew.sh/). Then, run:

```bash
brew install lapack openblas
```

Next, we must set the `LIBRARY_PATH`:

```bash
export LIBRARY_PATH=$LIBRARY_PATH:$(brew --prefix)/opt/lapack/lib:$(brew --prefix)/opt/openblas/lib
```

## <a name="threads"></a> Number of threads

By default, OpenBLAS will use all available threads, including Hyper-Threads that may worsen the performance. Thus, it is best to set the following environment variable:

```bash
export OPENBLAS_NUM_THREADS=<real-core-number>
```

Substitute `<real-core-number>` with the correct value from your system.

Furthermore, if working on a multi-threaded application where the solver should not be multi-threaded on its own (e.g., running parallel calculations in an optimization tool), you may set:

```bash
export OPENBLAS_NUM_THREADS=1
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
    //  . 4  2  .  1
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
    - [x] Add more complex number functions
    - [ ] Add fundamental functions to `russell_lab`
        - [ ] Implement the modified Bessel functions
    - [ ] Implement some numerical methods in `russell_lab`
        - [ ] Implement Brent's solver
        - [ ] Implement a solver for the cubic equation
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
    - [ ] Re-study the possibility of wrapping SuperLU (see deleted branch)
- [ ] Improve crate `russell_stat`
    - [x] Add probability distribution functions
    - [x] Implement drawing of ASCII histograms
- [ ] Improve the `russell_tensor` crate
    - [x] Implement functions to calculate invariants
    - [x] Implement first and second-order derivatives of invariants
    - [x] Implement some high-order derivatives
    - [ ] Implement standard continuum mechanics tensors
- [ ] Make it possible to install Russell on Windows and macOS 
    - [ ] Use Intel MKL on Windows
    - [ ] Install MUMPS and UMFPACK on Windows and macOS

## <a name="coverage"></a> Code coverage

### Sunburst

The innermost circle is the entire project; folders are moving away from the center, then a single file. Each slice's size and color represent the number of statements and the coverage, respectively.

![Sunburst](https://codecov.io/gh/cpmech/russell/graphs/sunburst.svg?token=PQWSKMZQXT)

### Grid

Each block represents a single file in the project. The size and color of each block are described by the number of statements and the coverage, respectively.

![Grid](https://codecov.io/gh/cpmech/russell/graphs/tree.svg?token=PQWSKMZQXT)

### Icicle

The top section represents the entire project. Proceeding with folders and, finally, individual files. Each slice's size and color define the number of statements and the coverage, respectively.

![Icicle](https://codecov.io/gh/cpmech/russell/graphs/icicle.svg?token=PQWSKMZQXT)
