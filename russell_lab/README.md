# Russell Lab - Matrix-vector laboratory including linear algebra tools

_This crate is part of [Russell - Rust Scientific Library](https://github.com/cpmech/russell)_

## Contents

* [Introduction](#introduction)
* [Installation on Debian/Ubuntu/Linux](#debian)
* [Installation on macOS](#macos)
* [Setting Cargo.toml](#cargo)
* [Examples](#examples)
* [About the column major representation](#col-major)
* [Benchmarks](#benchmarks)
* [For developers](#developers)

## <a name="introduction"></a> Introduction

This crate implements several functions to perform linear algebra computations--it is a **mat**rix-vector **lab**oratory 😉. We implement some functions in native Rust code as much as possible but also wrap the best tools available, such as [OpenBLAS](https://github.com/OpenMathLib/OpenBLAS) and [Intel MKL](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2023-2/overview.html).

The main structures are `NumVector` and `NumMatrix`, which are generic Vector and Matrix structures. The Matrix data is stored as [column-major](#col-major). The `Vector` and `Matrix` are `f64` and `Complex64` aliases of `NumVector` and `NumMatrix`, respectively.

The linear algebra functions currently handle only `(f64, i32)` pairs, i.e., accessing the `(double, int)` C functions. We also consider `(Complex64, i32)` pairs.

There are many functions for linear algebra, such as (for Real and Complex types):

* Vector addition, copy, inner and outer products, norms, and more
* Matrix addition, multiplication, copy, singular-value decomposition, eigenvalues, pseudo-inverse, inverse, norms, and more
* Matrix-vector multiplication, and more
* Solution of dense linear systems with symmetric or non-symmetric coefficient matrices, and more
* Reading writing files, `linspace`, grid generators, Stopwatch, linear fitting, and more
* Checking results, comparing float point numbers, and verifying the correctness of derivatives; see `russell_lab::check`

See the documentation for further information:

- [russell_lab documentation](https://docs.rs/russell_lab) - Contains the API reference and examples

## <a name="debian"></a> Installation on Debian/Ubuntu/Linux

`russell_lab` depends on an efficient BLAS library such as [OpenBLAS](https://github.com/OpenMathLib/OpenBLAS) and [Intel MKL](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2023-2/overview.html). Thus, we have two options:

1. Use the standard Debian packages based on OpenBLAS (default)
2. **(XOR)** Install Intel MKL, which includes LAPACK

Option 2 requires the following environment variable:

```bash
export RUSSELL_LAB_USE_INTEL_MKL=1
```

For convenience, you may use the scripts in the [zscripts](https://github.com/cpmech/russell/tree/main/russell_lab/zscripts) directory.

**1.** Use the standard Debian packages based on OpenBLAS:

```bash
bash zscripts/01-ubuntu-openblas.bash
```

**2.** Install Intel MKL:

```bash
bash zscripts/02-ubuntu-intel-mkl.bash
```

### Number of threads

By default, OpenBLAS and intel MKL may use all available threads, including "hyper-threads." If desirable, you may set the allowed number of threads with the following environment variable:

```bash
export OPENBLAS_NUM_THREADS=1
``` 

## <a name="macos"></a> Installation on macOS

At this time, only OpenBLAS has been tested on macOS.

First, install [Homebrew](https://brew.sh/). Then, run:

```bash
brew install lapack openblas
```

Next, we must set the `LIBRARY_PATH`:

```bash
export LIBRARY_PATH=$LIBRARY_PATH:$(brew --prefix)/opt/lapack/lib:$(brew --prefix)/opt/openblas/lib
```

## <a name="cargo"></a> Setting Cargo.toml

[![Crates.io](https://img.shields.io/crates/v/russell_lab.svg)](https://crates.io/crates/russell_lab)

👆 Check the crate version and update your Cargo.toml accordingly:

```toml
[dependencies]
russell_lab = "*"
```

## <a name="examples"></a> Examples

See also:

* [russell_lab/examples](https://github.com/cpmech/russell/tree/main/russell_lab/examples)

### Compute the pseudo-inverse matrix

```rust
use russell_lab::{mat_pseudo_inverse, Matrix, StrError};

fn main() -> Result<(), StrError> {
    // set matrix
    let mut a = Matrix::from(&[
      [1.0, 0.0], //
      [0.0, 1.0], //
      [0.0, 1.0], //
    ]);
    let a_copy = a.clone();

    // compute pseudo-inverse matrix (because it's square)
    let mut ai = Matrix::new(2, 3);
    mat_pseudo_inverse(&mut ai, &mut a)?;

    // compare with solution
    let ai_correct = "┌                ┐\n\
                      │ 1.00 0.00 0.00 │\n\
                      │ 0.00 0.50 0.50 │\n\
                      └                ┘";
    assert_eq!(format!("{:.2}", ai), ai_correct);

    // compute a ⋅ ai
    let (m, n) = a.dims();
    let mut a_ai = Matrix::new(m, m);
    for i in 0..m {
        for j in 0..m {
            for k in 0..n {
                a_ai.add(i, j, a_copy.get(i, k) * ai.get(k, j));
            }
        }
    }

    // check: a ⋅ ai ⋅ a = a
    let mut a_ai_a = Matrix::new(m, n);
    for i in 0..m {
        for j in 0..n {
            for k in 0..m {
                a_ai_a.add(i, j, a_ai.get(i, k) * a_copy.get(k, j));
            }
        }
    }
    let a_ai_a_correct = "┌           ┐\n\
                          │ 1.00 0.00 │\n\
                          │ 0.00 1.00 │\n\
                          │ 0.00 1.00 │\n\
                          └           ┘";
    assert_eq!(format!("{:.2}", a_ai_a), a_ai_a_correct);
    Ok(())
}
```

### Compute eigenvalues

```rust
use russell_lab::*;

fn main() -> Result<(), StrError> {
    // set matrix
    let data = [[2.0, 0.0, 0.0], [0.0, 3.0, 4.0], [0.0, 4.0, 9.0]];
    let mut a = Matrix::from(&data);

    // allocate output arrays
    let m = a.nrow();
    let mut l_real = Vector::new(m);
    let mut l_imag = Vector::new(m);
    let mut v_real = Matrix::new(m, m);
    let mut v_imag = Matrix::new(m, m);

    // perform the eigen-decomposition
    mat_eigen(&mut l_real, &mut l_imag, &mut v_real, &mut v_imag, &mut a)?;

    // check results
    assert_eq!(
        format!("{:.1}", l_real),
        "┌      ┐\n\
         │ 11.0 │\n\
         │  1.0 │\n\
         │  2.0 │\n\
         └      ┘"
    );
    assert_eq!(
        format!("{}", l_imag),
        "┌   ┐\n\
         │ 0 │\n\
         │ 0 │\n\
         │ 0 │\n\
         └   ┘"
    );

    // check eigen-decomposition (similarity transformation) of a
    // symmetric matrix with real-only eigenvalues and eigenvectors
    let a_copy = Matrix::from(&data);
    let lam = Matrix::diagonal(l_real.as_data());
    let mut a_v = Matrix::new(m, m);
    let mut v_l = Matrix::new(m, m);
    let mut err = Matrix::filled(m, m, f64::MAX);
    mat_mat_mul(&mut a_v, 1.0, &a_copy, &v_real)?;
    mat_mat_mul(&mut v_l, 1.0, &v_real, &lam)?;
    mat_add(&mut err, 1.0, &a_v, -1.0, &v_l)?;
    approx_eq(mat_norm(&err, Norm::Max), 0.0, 1e-15);
    Ok(())
}
```

### Cholesky factorization

```rust
use russell_lab::*;

fn main() -> Result<(), StrError> {
    // set matrix
    let sym = 0.0;
    #[rustfmt::skip]
    let mut a = Matrix::from(&[
        [  4.0,   sym,   sym],
        [ 12.0,  37.0,   sym],
        [-16.0, -43.0,  98.0],
    ]);

    // perform factorization
    mat_cholesky(&mut a, false)?;

    // define alias (for convenience)
    let l = &a;

    // compare with solution
    let l_correct = "┌          ┐\n\
                     │  2  0  0 │\n\
                     │  6  1  0 │\n\
                     │ -8  5  3 │\n\
                     └          ┘";
    assert_eq!(format!("{}", l), l_correct);

    // check:  l ⋅ lᵀ = a
    let m = a.nrow();
    let mut l_lt = Matrix::new(m, m);
    for i in 0..m {
        for j in 0..m {
            for k in 0..m {
                l_lt.add(i, j, l.get(i, k) * l.get(j, k));
            }
        }
    }
    let l_lt_correct = "┌             ┐\n\
                        │   4  12 -16 │\n\
                        │  12  37 -43 │\n\
                        │ -16 -43  98 │\n\
                        └             ┘";
    assert_eq!(format!("{}", l_lt), l_lt_correct);
    Ok(())
}
```

## <a name="col-major"></a> About the column major representation

Only the COL-MAJOR representation is considered here.

```text
    ┌     ┐  row_major = {0, 3,
    │ 0 3 │               1, 4,
A = │ 1 4 │               2, 5};
    │ 2 5 │
    └     ┘  col_major = {0, 1, 2,
    (m × n)               3, 4, 5}

Aᵢⱼ = col_major[i + j·m] = row_major[i·n + j]
        ↑
COL-MAJOR IS ADOPTED HERE
```

The main reason to use the **col-major** representation is to make the code work better with BLAS/LAPACK written in Fortran. Although those libraries have functions to handle row-major data, they usually add an overhead due to temporary memory allocation and copies, including transposing matrices. Moreover, the row-major versions of some BLAS/LAPACK libraries produce incorrect results (notably the DSYEV).

## <a name="benchmarks"></a> Benchmarks

Need to install:

```bash
cargo install cargo-criterion
```

Run the benchmarks with:

```bash
bash ./zscripts/benchmark.bash
```

### Jacobi Rotation versus LAPACK DSYEV

Comparison of the performances of `mat_eigen_sym_jacobi` (Jacobi rotation) versus `mat_eigen_sym` (calling LAPACK DSYEV).

![Jacobi Rotation versus LAPACK DSYEV (1-5)](data/figures/bench_mat_eigen_sym_1-5.svg)

![Jacobi Rotation versus LAPACK DSYEV (1-32)](data/figures/bench_mat_eigen_sym_1-32.svg)

## <a name="developers"></a> For developers

Notes for developers:

* The `c_code` directory contains a thin wrapper to the BLAS libraries (OpenBLAS or Intel MKL)
* The `c_code` directory also contains a wrapper to the C math functions
* The `build.rs` file uses the crate `cc` to build the C-wrappers
* The `zscripts` directory contains Bash scripts to install the Intel MKL on Debian/Linux
