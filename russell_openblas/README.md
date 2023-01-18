# Russell OpenBLAS - Thin wrapper to some OpenBLAS routines

_This crate is part of [Russell - Rust Scientific Library](https://github.com/cpmech/russell)_

This package implements a thin wrapper to a few of the OpenBLAS routines for performing linear algebra computations.

Documentation:

- [API reference (docs.rs)](https://docs.rs/russell_openblas)

## Installation

Install some libraries:

```bash
sudo apt-get install \
    liblapacke-dev \
    libopenblas-dev
```

[![Crates.io](https://img.shields.io/crates/v/russell_openblas.svg)](https://crates.io/crates/russell_openblas)

👆 Check the crate version and update your Cargo.toml accordingly:

```toml
[dependencies]
russell_openblas = "*"
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

### Vector operations

```rust
use russell_openblas::{dcopy, ddot};

fn main() {
    // ddot
    let u = [1.0, 2.0, 3.0, 4.0];
    let v = [4.0, 3.0, 2.0, 1.0];
    assert_eq!(ddot(4, &u, 1, &v, 1), 20.0);

    // dcopy
    let mut w = vec![0.0; 4];
    dcopy(4, &u, 1, &mut w, 1);
    assert_eq!(w, &[1.0, 2.0, 3.0, 4.0]);
}
```

### Matrix multiplication

```rust
use russell_chk::vec_approx_eq;
use russell_openblas::{col_major, dgemm};

fn main() {
    // 0.5⋅a⋅b + 2⋅c

    // allocate matrices
    let a = col_major(4, 5, &[ // (m, k) = (4, 5)
        1.0, 2.0,  0.0, 1.0, -1.0,
        2.0, 3.0, -1.0, 1.0,  1.0,
        1.0, 2.0,  0.0, 4.0, -1.0,
        4.0, 0.0,  3.0, 1.0,  1.0,
    ]);
    let b = col_major(5, 3, &[ // (k, n) = (5, 3)
        1.0, 0.0, 0.0,
        0.0, 0.0, 3.0,
        0.0, 0.0, 1.0,
        1.0, 0.0, 1.0,
        0.0, 2.0, 0.0,
    ]);
    let mut c = col_major(4, 3, &[ // (m, n) = (4, 3)
          0.50, 0.0,  0.25,
          0.25, 0.0, -0.25,
        -0.25, 0.0,  0.00,
        -0.25, 0.0,  0.00,
    ]);

    // sizes
    let m = 4; // m = nrow(a) = a.M = nrow(c)
    let k = 5; // k = ncol(a) = a.N = nrow(b)
    let n = 3; // n = ncol(b) = b.N = ncol(c)

    // run dgemm
    let (trans_a, trans_b) = (false, false);
    let (alpha, beta) = (0.5, 2.0);
    dgemm(trans_a, trans_b, m, n, k, alpha, &a, &b, beta, &mut c);

    // check
    let correct = col_major(4, 3, &[
        2.0, -1.0, 4.0,
        2.0,  1.0, 4.0,
        2.0, -1.0, 5.0,
        2.0,  1.0, 2.0,
    ]);
    vec_approx_eq(&c, &correct, 1e-15);
}
```

### Solution of linear system

```rust
use russell_chk::vec_approx_eq;
use russell_openblas::{col_major, dgesv, StrError};

fn main() -> Result<(), StrError> {
    // matrix
    let mut a = col_major(5, 5, &[
        2.0,  3.0,  0.0, 0.0, 0.0,
        3.0,  0.0,  4.0, 0.0, 6.0,
        0.0, -1.0, -3.0, 2.0, 0.0,
        0.0,  0.0,  1.0, 0.0, 0.0,
        0.0,  4.0,  2.0, 0.0, 1.0,
    ]);

    // right-hand-side
    let mut b = vec![8.0, 45.0, -3.0, 3.0, 19.0];

    // solve b := x := A⁻¹ b
    let (n, nrhs) = (5_i32, 1_i32);
    let mut ipiv = vec![0; n as usize];
    dgesv(n, nrhs, &mut a, &mut ipiv, &mut b)?;

    // check
    let correct = &[1.0, 2.0, 3.0, 4.0, 5.0];
    vec_approx_eq(&b, correct, 1e-14);
    Ok(())
}
```
