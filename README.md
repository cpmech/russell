# Russell - Rust Scientific Library

[![codecov](https://codecov.io/gh/cpmech/russell/graph/badge.svg?token=PQWSKMZQXT)](https://codecov.io/gh/cpmech/russell)

![Bertrand Russell](zassets/Bertrand_Russell_1957.jpg)

([CC0](http://creativecommons.org/publicdomain/zero/1.0/deed.en). Photo: [Bertrand Russell](https://en.wikipedia.org/wiki/Bertrand_Russell))

**Russell** assists in the development of scientific computations using the Rust language. We focus on numerical methods and solvers for differential equations; however, anything is possible 😉.

An essential goal of this library is to bring the best (fastest) solutions while maintaining a very **clean** (and idiomatic) code, thoroughly tested (min coverage of 95%; see Appendix B below), and yet simple to use. The best solutions are brought by wrapping **powerful** libraries such as OpenBLAS, MUMPS, and SuiteSparse (UMFPACK).

Available crates:

- [![Crates.io](https://img.shields.io/crates/v/russell_chk.svg)](https://crates.io/crates/russell_chk) [chk](https://github.com/cpmech/russell/tree/main/russell_chk) Functions to check vectors and other data in tests
- [![Crates.io](https://img.shields.io/crates/v/russell_lab.svg)](https://crates.io/crates/russell_lab) [lab](https://github.com/cpmech/russell/tree/main/russell_lab) Matrix-vector laboratory including linear algebra tools
- [![Crates.io](https://img.shields.io/crates/v/russell_openblas.svg)](https://crates.io/crates/russell_openblas) [openblas](https://github.com/cpmech/russell/tree/main/russell_openblas) Thin wrapper to some OpenBLAS routines
- [![Crates.io](https://img.shields.io/crates/v/russell_sparse.svg)](https://crates.io/crates/russell_sparse) [sparse](https://github.com/cpmech/russell/tree/main/russell_sparse) Sparse matrix tools and solvers
- [![Crates.io](https://img.shields.io/crates/v/russell_stat.svg)](https://crates.io/crates/russell_stat) [stat](https://github.com/cpmech/russell/tree/main/russell_stat) Statistics calculations, probability distributions, and pseudo random numbers
- [![Crates.io](https://img.shields.io/crates/v/russell_tensor.svg)](https://crates.io/crates/russell_tensor) [tensor](https://github.com/cpmech/russell/tree/main/russell_tensor) Tensor analysis structures and functions for continuum mechanics

External recommended crate:

- [plotpy](https://github.com/cpmech/plotpy) Plotting tools using Python3/Matplotlib as an engine

## Examples

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

    // define correct data
    let s_correct = "┌      ┐\n\
                     │ 5.46 │\n\
                     │ 0.37 │\n\
                     └      ┘";
    let u_correct = "┌                         ┐\n\
                     │ -0.82 -0.58  0.00  0.00 │\n\
                     │ -0.58  0.82  0.00  0.00 │\n\
                     │  0.00  0.00  1.00  0.00 │\n\
                     │  0.00  0.00  0.00  1.00 │\n\
                     └                         ┘";
    let vt_correct = "┌             ┐\n\
                      │ -0.40 -0.91 │\n\
                      │ -0.91  0.40 │\n\
                      └             ┘";

    // check solution
    assert_eq!(format!("{:.2}", s), s_correct);
    assert_eq!(format!("{:.2}", u), u_correct);
    assert_eq!(format!("{:.2}", vt), vt_correct);

    // check SVD: a == u * s * vt
    let mut usv = Matrix::new(m, n);
    for i in 0..m {
        for j in 0..n {
            for k in 0..min_mn {
                usv.add(i, j, u.get(i, k) * s[k] * vt.get(k, j));
            }
        }
    }
    let usv_correct = "┌     ┐\n\
                       │ 2 4 │\n\
                       │ 1 3 │\n\
                       │ 0 0 │\n\
                       │ 0 0 │\n\
                       └     ┘";
    assert_eq!(format!("{}", usv), usv_correct);
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

TODO

## Todo list

- [x] Add complex numbers functions to `russell_openblas`
- [ ] Add more complex numbers functions to `russell_lab`
- [ ] Add fundamental functions to `russell_lab`
    - [ ] Implement the modified Bessel functions
- [ ] Implement some numerical methods in `russell_lab`
    - [ ] Implement Brent's solver
    - [ ] Implement solver for the cubic equation
    - [ ] Implement numerical derivation
    - [ ] Implement numerical Jacobian function
    - [ ] Implement Newton's method for nonlinear systems
    - [ ] Implement numerical quadrature
- [ ] Add interpolation and polynomials to `russell_lab`
    - [ ] Implement Chebyshev interpolation and polynomials
    - [ ] Implement Orthogonal polynomials
    - [ ] Implement Lagrange interpolation
- [x] Improve the `russell_sparse` crate
    - [x] Implement the Compressed Sparse Column format (CSC)
    - [x] Implement the Compressed Sparse Row format (CSC)
    - [x] Improve the C-interface to UMFPACK and MUMPS
    - [x] Implement the C-interface to Intel DSS
    - [ ] Write the conversion from COO to CSC in Rust
    - [ ] Possibly re-write (after benchmarking) the conversion from COO to CSR
    - [ ] Re-study the possibility to wrap SuperLU (see deleted branch)
- [x] Add probability distribution functions to `russell_stat`
- [x] Finalize drawing of ASCII histogram in `russell_stat`
- [ ] Improve the `russell_tensor` crate
    - [x] Implement functions to calculate invariants
    - [x] Implement first and second order derivatives of invariants
    - [x] Implement some high-order derivatives
    - [ ] Implement standard continuum mechanics tensors
- [ ] Implement more integration tests for linear algebra
- [ ] Implement more examples
- [ ] Implement more benchmarks

## Appendix A -- Benchmarks

### Jacobi Rotation versus LAPACK DSYEV

Comparison of the performances of `mat_eigen_sym_jacobi` (Jacobi rotation) versus `mat_eigen_sym` (calling LAPACK DSYEV).

![Jacobi Rotation versus LAPACK DSYEV (1-5)](zassets/bench_mat_eigen_sym_1-5.svg)

![Jacobi Rotation versus LAPACK DSYEV (1-32)](zassets/bench_mat_eigen_sym_1-32.svg)

## Appendix B -- Code coverage

### Sunburst

The inner-most circle is the entire project, moving away from the center are folders then, finally, a single file. The size and color of each slice is representing the number of statements and the coverage, respectively.

![Sunburst](https://codecov.io/gh/cpmech/russell/graphs/sunburst.svg?token=PQWSKMZQXT)

### Grid

Each block represents a single file in the project. The size and color of each block is represented by the number of statements and the coverage, respectively.

![Grid](https://codecov.io/gh/cpmech/russell/graphs/tree.svg?token=PQWSKMZQXT)

### Icicle

The top section represents the entire project. Proceeding with folders and finally individual files. The size and color of each slice is representing the number of statements and the coverage, respectively.

![Icicle](https://codecov.io/gh/cpmech/russell/graphs/icicle.svg?token=PQWSKMZQXT)
