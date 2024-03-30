# Russell - Rust Scientific Library

[![codecov](https://codecov.io/gh/cpmech/russell/graph/badge.svg?token=PQWSKMZQXT)](https://codecov.io/gh/cpmech/russell)
[![Test & Coverage](https://github.com/cpmech/russell/actions/workflows/test_and_coverage.yml/badge.svg)](https://github.com/cpmech/russell/actions/workflows/test_and_coverage.yml)
[![Test with local libs](https://github.com/cpmech/russell/actions/workflows/test_with_local_libs.yml/badge.svg)](https://github.com/cpmech/russell/actions/workflows/test_with_local_libs.yml)
[![Test with Intel MKL](https://github.com/cpmech/russell/actions/workflows/test_with_intel_mkl.yml/badge.svg)](https://github.com/cpmech/russell/actions/workflows/test_with_intel_mkl.yml)

## Contents

* [Introduction](#introduction)
* [Crates](#crates)
* [Installation on Debian/Ubuntu/Linux](#installation)
* [Installation on macOS](#macos)
* [Number of threads](#threads)
* [Examples](#examples)
    * [(lab) Singular value decomposition](#svd)
    * [(lab) Cholesky factorization](#cholesky)
    * [(lab) Solve a tiny (dense) linear system](#dense-lin-sys)
    * [(sparse) Solve a small sparse linear system](#sparse-lin-sys)
    * [(ode) Solve the brusselator ODE system](#brusselator)
    * [(stat) Generate the Frechet distribution](#frechet)
    * [(tensor) Allocate second-order tensors](#tensor)
* [Todo list](#todo)

<a name="introduction"></a>

## Introduction

**Russell** (Rust Scientific Library) assists in developing scientific computations using the Rust language.

The "main" crate here is [russell_lab](https://github.com/cpmech/russell/tree/main/russell_lab), a **mat**rix-vector **lab**oratory, which provides the fundamental `Vector` and `Matrix` structures and several functions to perform linear algebra computations. Thus, we recommend looking at [russell_lab](https://github.com/cpmech/russell/tree/main/russell_lab) first.

Next, we recommend looking at the [russell_sparse](https://github.com/cpmech/russell/tree/main/russell_sparse) crate, which implements sparse matrix structures such as COO (coordinates), CSC (compressed sparse column), and CSR (compressed sparse row) formats. `russell_sparse` also wraps powerful linear system solvers such as [UMFPACK](https://github.com/DrTimothyAldenDavis/SuiteSparse) and [MUMPS](https://mumps-solver.org).

<a name="crates"></a> 

## Crates

Available crates:

- [![Crates.io](https://img.shields.io/crates/v/russell_lab.svg)](https://crates.io/crates/russell_lab) [russell_lab](https://github.com/cpmech/russell/tree/main/russell_lab) Matrix-vector laboratory for linear algebra (with OpenBLAS or Intel MKL)
- [![Crates.io](https://img.shields.io/crates/v/russell_sparse.svg)](https://crates.io/crates/russell_sparse) [russell_sparse](https://github.com/cpmech/russell/tree/main/russell_sparse) Sparse matrix tools and solvers (with MUMPS and UMFPACK)
- [![Crates.io](https://img.shields.io/crates/v/russell_ode.svg)](https://crates.io/crates/russell_ode) [russell_ode](https://github.com/cpmech/russell/tree/main/russell_ode) Solvers for Ordinary differential equations (ODEs) and differential algebraic equations (DAEs) 
- [![Crates.io](https://img.shields.io/crates/v/russell_stat.svg)](https://crates.io/crates/russell_stat) [russell_stat](https://github.com/cpmech/russell/tree/main/russell_stat) Statistics calculations, probability distributions, and pseudo random numbers
- [![Crates.io](https://img.shields.io/crates/v/russell_tensor.svg)](https://crates.io/crates/russell_tensor) [russell_tensor](https://github.com/cpmech/russell/tree/main/russell_tensor) Tensor analysis structures and functions for continuum mechanics


👆 Check the crate version and update your Cargo.toml accordingly. Examples:

```toml
[dependencies]
russell_lab = "*"
russell_sparse = "*"
russell_ode = "*"
russell_stat = "*"
russell_tensor = "*"
```

Or, considering the optional _features_ (see more about these next):

```toml
[dependencies]
russell_lab = { version = "*", features = ["intel_mkl"] }
russell_sparse = { version = "*", features = ["local_libs", "intel_mkl"] }
russell_ode = { version = "*", features = ["intel_mkl"] }
russell_stat = { version = "*", features = ["intel_mkl"] }
russell_tensor = { version = "*", features = ["intel_mkl"] }
```

External associated and recommended crates:

- [plotpy](https://github.com/cpmech/plotpy) Plotting tools using Python3/Matplotlib as an engine (for quality graphics)
- [tritet](https://github.com/cpmech/tritet) Triangle and tetrahedron mesh generators (with Triangle and Tetgen)
- [gemlab](https://github.com/cpmech/gemlab) Geometry, meshes, and numerical integration for finite element analyses



<a name="installation"></a>

## Installation on Debian/Ubuntu/Linux

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

<a name="macos"></a>

## Installation on macOS

Currently, only OpenBLAS has been tested on macOS.

First, install [Homebrew](https://brew.sh/). Then, run:

```bash
brew install lapack openblas
```

Next, we must set the `LIBRARY_PATH`:

```bash
export LIBRARY_PATH=$LIBRARY_PATH:$(brew --prefix)/opt/lapack/lib:$(brew --prefix)/opt/openblas/lib
```

<a name="threads"></a>

## Number of threads

By default, OpenBLAS will use all available threads, including Hyper-Threads that may worsen the performance. Thus, it is best to set the following environment variable:

```bash
export OPENBLAS_NUM_THREADS=<real-core-number>
```

Substitute `<real-core-number>` with the correct value from your system.

Furthermore, if working on a multi-threaded application where the solver should not be multi-threaded on its own (e.g., running parallel calculations in an optimization tool), you may set:

```bash
export OPENBLAS_NUM_THREADS=1
```

<a name="examples"></a>

## Examples

See also:

* [russell_lab/examples](https://github.com/cpmech/russell/tree/main/russell_lab/examples)
* [russell_sparse/examples](https://github.com/cpmech/russell/tree/main/russell_sparse/examples)
* [russell_ode/examples](https://github.com/cpmech/russell/tree/main/russell_ode/examples)
* [russell_stat/examples](https://github.com/cpmech/russell/tree/main/russell_stat/examples)
* [russell_tensor/examples](https://github.com/cpmech/russell/tree/main/russell_tensor/examples)

**Note:** For the functions dealing with complex numbers, the following line must be added to all derived code:

```rust
use num_complex::Complex64;
```

This line will bring `Complex64` to the scope. For convenience the (russell_lab) macro `cpx!` may be used to allocate complex numbers.

<a name="svd"></a>

### (lab) Singular value decomposition

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

<a name="cholesky"></a>

### (lab) Cholesky factorization

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

<a name="dense-lin-sys"></a>

### (lab) Solve a tiny (dense) linear system

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

### <a name="sparse-lin-sys"></a> (sparse) Solve a small sparse linear system using UMFPACK

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
    let mut coo = SparseMatrix::new_coo(ndim, ndim, nnz, Sym::No)?;
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
    let (mx, ex) = (stats.determinant.mantissa_real, stats.determinant.exponent);
    println!("det(a) = {:?}", mx * f64::powf(10.0, ex));
    println!("rcond  = {:?}", stats.output.umfpack_rcond_estimate);
    Ok(())
}
```

<a name="brusselator"></a>

### (ode) Solve the brusselator ODE system

The system is:

```text
y0' = 1 - 4 y0 + y0² y1
y1' = 3 y0 - y0² y1

with  y0(x=0) = 3/2  and  y1(x=0) = 3
```

Solving with DoPri8 -- 8(5,3):

```rust
use russell_lab::StrError;
use russell_ode::prelude::*;

fn main() -> Result<(), StrError> {
    // get the ODE system
    let (system, x0, y0, mut args, y_ref) = Samples::brusselator_ode();

    // solver
    let params = Params::new(Method::DoPri8);
    let mut solver = OdeSolver::new(params, &system)?;

    // enable dense output
    let mut out = Output::new();
    let h_out = 0.01;
    let selected_y_components = &[0, 1];
    out.set_dense_recording(true, h_out, selected_y_components)?;

    // solve the problem
    solver.solve(&mut y0, x0, data.x1, None, Some(&mut out), &mut args)?;

    // print the results and stats
    println!("y_russell     = {:?}", y0.as_data());
    println!("y_mathematica = {:?}", y_ref.as_data());
    println!("{}", solver.stats());
    Ok(())
}
```

A plot of the (dense) solution is shown below:

![Brusselator results: DoPri8](russell_ode/data/figures/brusselator_dopri8.svg)

<a name="frechet"></a>

### (stat) Generate the Frechet distribution

Code:

```rust
use russell_stat::*;

fn main() -> Result<(), StrError> {
    // generate samples
    let mut rng = rand::thread_rng();
    let dist = DistributionFrechet::new(0.0, 1.0, 1.0)?;
    let nsamples = 10_000;
    let mut data = vec![0.0; nsamples];
    for i in 0..nsamples {
        data[i] = dist.sample(&mut rng);
    }
    println!("{}", statistics(&data));

    // text-plot
    let stations = (0..20).map(|i| (i as f64) * 0.5).collect::<Vec<f64>>();
    let mut hist = Histogram::new(&stations)?;
    hist.count(&data);
    println!("{}", hist);
    Ok(())
}
```

Sample output:

```text
min = 0.11845731988882305
max = 26248.036672205748
mean = 12.268212841918867
std_dev = 312.7131690782321

[  0,0.5) | 1370 🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦
[0.5,  1) | 2313 🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦
[  1,1.5) | 1451 🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦
[1.5,  2) |  971 🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦
[  2,2.5) |  659 🟦🟦🟦🟦🟦🟦🟦🟦
[2.5,  3) |  460 🟦🟦🟦🟦🟦
[  3,3.5) |  345 🟦🟦🟦🟦
[3.5,  4) |  244 🟦🟦🟦
[  4,4.5) |  216 🟦🟦
[4.5,  5) |  184 🟦🟦
[  5,5.5) |  133 🟦
[5.5,  6) |  130 🟦
[  6,6.5) |  115 🟦
[6.5,  7) |  108 🟦
[  7,7.5) |   70 
[7.5,  8) |   75 
[  8,8.5) |   57 
[8.5,  9) |   48 
[  9,9.5) |   59 
      sum = 9008
```

<a name="tensor"></a>

### (tensor) Allocate second-order tensors

```rust
use russell_tensor::*;

fn main() -> Result<(), StrError> {
    // general
    let a = Tensor2::from_matrix(
        &[
            [1.0, SQRT_2 * 2.0, SQRT_2 * 3.0],
            [SQRT_2 * 4.0, 5.0, SQRT_2 * 6.0],
            [SQRT_2 * 7.0, SQRT_2 * 8.0, 9.0],
        ],
        Mandel::General,
    )?;
    assert_eq!(
        format!("{:.1}", a.vec),
        "┌      ┐\n\
         │  1.0 │\n\
         │  5.0 │\n\
         │  9.0 │\n\
         │  6.0 │\n\
         │ 14.0 │\n\
         │ 10.0 │\n\
         │ -2.0 │\n\
         │ -2.0 │\n\
         │ -4.0 │\n\
         └      ┘"
    );

    // symmetric-3D
    let b = Tensor2::from_matrix(
        &[
            [1.0, 4.0 / SQRT_2, 6.0 / SQRT_2],
            [4.0 / SQRT_2, 2.0, 5.0 / SQRT_2],
            [6.0 / SQRT_2, 5.0 / SQRT_2, 3.0],
        ],
        Mandel::Symmetric,
    )?;
    assert_eq!(
        format!("{:.1}", b.vec),
        "┌     ┐\n\
         │ 1.0 │\n\
         │ 2.0 │\n\
         │ 3.0 │\n\
         │ 4.0 │\n\
         │ 5.0 │\n\
         │ 6.0 │\n\
         └     ┘"
    );

    // symmetric-2D
    let c = Tensor2::from_matrix(
        &[[1.0, 4.0 / SQRT_2, 0.0], [4.0 / SQRT_2, 2.0, 0.0], [0.0, 0.0, 3.0]],
        Mandel::Symmetric2D,
    )?;
    assert_eq!(
        format!("{:.1}", c.vec),
        "┌     ┐\n\
         │ 1.0 │\n\
         │ 2.0 │\n\
         │ 3.0 │\n\
         │ 4.0 │\n\
         └     ┘"
    );
    Ok(())
}
```

<a name="todo"></a>

## Todo list

- [ ] Improve `russell_lab`
    - [x] Implement more integration tests for linear algebra
    - [x] Implement more examples
    - [ ] Implement more benchmarks
    - [x] Wrap more BLAS/LAPACK functions
        - [x] Implement dggev, zggev, zheev, and zgeev
    - [x] Wrap Intel MKL (option for OpenBLAS)
    - [x] Add more complex number functions
    - [ ] Add fundamental functions to `russell_lab`
        - [ ] Implement the modified Bessel functions
    - [ ] Implement some numerical methods in `russell_lab`
        - [ ] Implement Brent's solver
        - [ ] Implement a solver for the cubic equation
        - [x] Implement numerical derivation
        - [ ] Implement numerical Jacobian function
        - [ ] Implement Newton's method for nonlinear systems
        - [ ] Implement numerical quadrature
    - [ ] Add interpolation and polynomials to `russell_lab`
        - [ ] Implement Chebyshev interpolation and polynomials
        - [ ] Implement Orthogonal polynomials
        - [ ] Implement Lagrange interpolation
    - [x] Implement FFT
        - [x] Partially wrap FFTW (with warnings about it being thread-unsafe)
- [x] Improve `russell_sparse`
    - [x] Wrap the KLU solver (in addition to MUMPS and UMFPACK)
    - [x] Implement the Compressed Sparse Column format (CSC)
    - [x] Implement the Compressed Sparse Row format (CSC)
    - [x] Improve the C-interface to UMFPACK and MUMPS
    - [x] Write the conversion from COO to CSC in Rust
- [x] Improve `russell_ode`
    - [x] Implement explicit Runge-Kutta solvers
    - [x] Implement Radau5 for DAEs
- [ ] Improve `russell_stat`
    - [x] Add probability distribution functions
    - [x] Implement drawing of ASCII histograms
    - [ ] Add more examples
- [ ] Improve `russell_tensor`
    - [x] Implement functions to calculate invariants
    - [x] Implement first and second-order derivatives of invariants
    - [x] Implement some high-order derivatives
    - [ ] Implement standard continuum mechanics tensors
- [ ] Study the possibility to install Russell on Windows and macOS 
    - [ ] Use Intel MKL on Windows
    - [ ] Install MUMPS and UMFPACK on Windows and macOS
