# Russell Lab - Matrix-vector laboratory including linear algebra tools

_This crate is part of [Russell - Rust Scientific Library](https://github.com/cpmech/russell)_

This repository is a "rust laboratory" for vectors and matrices.

Documentation:

- [API reference (docs.rs)](https://docs.rs/russell_lab)

## Installation

Install some libraries:

```bash
sudo apt-get install \
    liblapacke-dev \
    libopenblas-dev
```

Add this to your Cargo.toml (choose the right version):

```toml
[dependencies]
russell_lab = "*"
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

### Compute the pseudo-inverse matrix

```rust
use russell_lab::{pseudo_inverse, Matrix, StrError};

fn main() -> Result<(), StrError> {
    // set matrix
    let mut a = Matrix::from(&[
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 1.0],
    ]);
    let a_copy = a.get_copy();

    // compute pseudo-inverse matrix (because it's square)
    let mut ai = Matrix::new(2, 3);
    pseudo_inverse(&mut ai, &mut a)?;

    // compare with solution
    let ai_correct = "┌                ┐\n\
                      │ 1.00 0.00 0.00 │\n\
                      │ 0.00 0.50 0.50 │\n\
                      └                ┘";
    assert_eq!(format!("{:.2}", ai), ai_correct);

    // compute a⋅ai
    let (m, n) = a.dims();
    let mut a_ai = Matrix::new(m, m);
    for i in 0..m {
        for j in 0..m {
            for k in 0..n {
                a_ai[i][j] += a_copy[i][k] * ai[k][j];
            }
        }
    }

    // check if a⋅ai⋅a == a
    let mut a_ai_a = Matrix::new(m, n);
    for i in 0..m {
        for j in 0..n {
            for k in 0..m {
                a_ai_a[i][j] += a_ai[i][k] * a_copy[k][j];
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
use russell_lab::{add_matrices, eigen_decomp, mat_mat_mul, matrix_norm, NormMat, Matrix, StrError};
use russell_chk::assert_approx_eq;

fn main() -> Result<(), StrError> {
    // set matrix
    let data = [
        [2.0, 0.0, 0.0],
        [0.0, 3.0, 4.0],
        [0.0, 4.0, 9.0],
    ];
    let mut a = Matrix::from(&data);

    // allocate output arrays
    let m = a.nrow();
    let mut l_real = vec![0.0; m];
    let mut l_imag = vec![0.0; m];
    let mut v_real = Matrix::new(m, m);
    let mut v_imag = Matrix::new(m, m);

    // perform the eigen-decomposition
    eigen_decomp(
        &mut l_real,
        &mut l_imag,
        &mut v_real,
        &mut v_imag,
        &mut a,
    )?;

    // check results
    let l_real_correct = "[11.0, 1.0, 2.0]";
    let l_imag_correct = "[0.0, 0.0, 0.0]";
    let v_real_correct = "┌                      ┐\n\
                          │  0.000  0.000  1.000 │\n\
                          │  0.447  0.894  0.000 │\n\
                          │  0.894 -0.447  0.000 │\n\
                          └                      ┘";
    let v_imag_correct = "┌       ┐\n\
                          │ 0 0 0 │\n\
                          │ 0 0 0 │\n\
                          │ 0 0 0 │\n\
                          └       ┘";
    assert_eq!(format!("{:?}", l_real), l_real_correct);
    assert_eq!(format!("{:?}", l_imag), l_imag_correct);
    assert_eq!(format!("{:.3}", v_real), v_real_correct);
    assert_eq!(format!("{}", v_imag), v_imag_correct);

    // check eigen-decomposition (similarity transformation) of a
    // symmetric matrix with real-only eigenvalues and eigenvectors
    let a_copy = Matrix::from(&data);
    let lam = Matrix::diagonal(&l_real);
    let mut a_v = Matrix::new(m, m);
    let mut v_l = Matrix::new(m, m);
    let mut err = Matrix::filled(m, m, f64::MAX);
    mat_mat_mul(&mut a_v, 1.0, &a_copy, &v_real)?;
    mat_mat_mul(&mut v_l, 1.0, &v_real, &lam)?;
    add_matrices(&mut err, 1.0, &a_v, -1.0, &v_l)?;
    assert_approx_eq!(matrix_norm(&err, NormMat::Max), 0.0, 1e-15);
    Ok(())
}
```
