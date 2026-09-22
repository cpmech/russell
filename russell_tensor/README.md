# Russell Tensor - Tensor analysis, calculus, and functions for continuum mechanics <!-- omit from toc --> 

[![documentation](https://docs.rs/russell_tensor/badge.svg)](https://docs.rs/russell_tensor/)

_This crate is part of [Russell - Rust Scientific Library](https://github.com/cpmech/russell)_

## Contents <!-- omit from toc --> 

- [Introduction](#introduction)
  - [Capabilities](#capabilities)
  - [Kelvin-Mandel notation](#kelvin-mandel-notation)
  - [Documentation](#documentation)
- [Installation](#installation)
  - [Setting Cargo.toml](#setting-cargotoml)
  - [Optional features](#optional-features)
- [🌟 Examples](#-examples)
  - [Computing the Invariants](#computing-the-invariants)
  - [Allocating Second Order Tensors](#allocating-second-order-tensors)
- [For developers](#for-developers)
- [Principal invariants (symmetric)](#principal-invariants-symmetric)



## Introduction

This library implements structures and functions for tensor analysis and calculus, with focus on applications in engineering and [Continuum Mechanics](https://en.wikipedia.org/wiki/Continuum_mechanics). The essential functionality for the targeted applications includes first-order, second-order, third-order, and fourth-order tensors, scalar "invariants," and derivatives.

### Capabilities

* `Tensor1` — first-order tensors (vectors in R3) with operations such as the dot and cross products
* `Tensor2` — second-order tensors (symmetric or not) with functions such as the determinant, inverse, norm, and invariants (principal, deviatoric, Lode, octahedral, ...)
* `Tensor3` — third-order tensors (minor-symmetric or not)
* `Tensor4` — fourth-order tensors (minor-symmetric or not)
* Operations between tensors — addition, single and double contractions (dot and ddot), and dyadic products; most operations support both overwriting (`SET`) and accumulation (`ADD`)
* Analytical derivatives — first and second derivatives of invariants and tensor functions (e.g., the inverse and squared tensors) with respect to tensors
* `EigenValuesT2`, `EigenProjsT2`, `EigenProjDerivsT2` — eigenvalues, eigenprojectors, and the derivatives of the eigenprojectors of symmetric second-order tensors
* `LinElasticity` — the linear elasticity equations for small-strain problems (Hooke's law)
* `PiezoDatabase` — a database of piezoelectric materials (permittivity, piezoelectric, and stiffness tensors) loaded from JSON
* Constants — identity, transposition, and projector tensors, as well as the `ADD`/`SET` operation selectors
* Polar decomposition — `F = R U = V R` via the classic Eigen/SVD algorithms, Brannon's iterative algorithm, or the quaternion-based Higham & Noferini (2016) algorithm (`PolarAlgo`, `polar_decomp_mx`)

### Kelvin-Mandel notation

Internally, tensors are stored as vectors/matrices with components given with respect to the Kelvin-Mandel basis, i.e., the *Kelvin-Mandel* notation, a norm-preserving alternative to [Voigt notation](https://en.wikipedia.org/wiki/Voigt_notation). In the Kelvin-Mandel notation, a second-order tensor is mapped to a column matrix (vector), a third-order tensor is mapped to a rectangular matrix, and a fourth-order tensor is mapped to a square matrix. Factors such as `√2` multiply some components to yield the norm-preserving mapping.

The dimension — the const generic `N` of `Tensor2`/`Tensor4`, and `M`/`N` of `Tensor3` — selects the representation:

* `9` — all components (general): 9×1 / 9×3 / 3×9 / 9×9
* `6` — symmetric `Tensor2` / minor-symmetric `Tensor3`/`Tensor4` (3D): 6×1 / 6×3 / 3×6 / 6×6
* `4` — symmetric `Tensor2` / minor-symmetric `Tensor3`/`Tensor4` (2D): 4×1 / 4×3 / 3×4 / 4×4

The dimensions above correspond to `Tensor2` (vector), `Tensor3` (Case A / Case B rectangular matrix), and `Tensor4` (square matrix), respectively.

A `Tensor3` is stored as a rectangular Kelvin-Mandel matrix with dimensions `(M, N)` set by const generics. Two cases are considered, where `DIM` (the leading dimension) is one of 4, 6, or 9:

* **Case A** — `(DIM, 3)`, i.e. `M = DIM` and `N = 3`: the Tensor3 acts on a `Tensor1` (vector) yielding a `Tensor2` (`T = H · u`)
* **Case B** — `(3, DIM)`, i.e. `M = 3` and `N = DIM`: the Tensor3 acts on a `Tensor2` yielding a `Tensor1` (vector) (`v = M : S`)

For second-order tensors, the stored component order is:

| Representation     | Stored components                                                                                                               |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------- |
| `9` (general)      | `T11`, `T22`, `T33`, `(T12 + T21)/√2`, `(T23 + T32)/√2`, `(T13 + T31)/√2`, `(T12 - T21)/√2`, `(T23 - T32)/√2`, `(T13 - T31)/√2` |
| `6` (symmetric)    | `T11`, `T22`, `T33`, `√2 T12`, `√2 T23`, `√2 T13`                                                                               |
| `4` (symmetric 2D) | `T11`, `T22`, `T33`, `√2 T12`                                                                                                   |

Use the `*_std*` constructors and accessors when working with ordinary Cartesian
components, such as `Tensor2::from_std_matrix` and `Tensor2::get_std`. Use the
accessors without `std` only when working directly with the stored
Kelvin-Mandel components. For example, an off-diagonal component `T12 = 4`
is stored as `√2 × 4` in a symmetric tensor, so `from_std_matrix` expects `4`
while `get(3)` returns `√2 × 4`.

### Documentation

* [![documentation](https://docs.rs/russell_tensor/badge.svg)](https://docs.rs/russell_tensor/) — [russell_tensor documentation](https://docs.rs/russell_tensor/)



## Installation

This crate depends on `russell_lab`, which requires non-Rust high-performance libraries. [See the main README file for the steps to install these dependencies.](https://github.com/cpmech/russell)



### Setting Cargo.toml

[![Crates.io](https://img.shields.io/crates/v/russell_tensor.svg)](https://crates.io/crates/russell_tensor)

👆 Check the crate version and update your Cargo.toml accordingly:

```toml
[dependencies]
russell_tensor = "*"
```

### Optional features

The following (Rust) features are available:

* `intel_mkl`: Use Intel MKL instead of OpenBLAS
* `heap`: Use heap-allocated (dynamically allocated) storage for the tensor components instead of the default stack-allocated (fixed-size) storage

Note that the [main README file](https://github.com/cpmech/russell) presents the steps to compile the required libraries according to each feature.



## 🌟 Examples

This section illustrates how to use `russell_tensor`. See also:

* [More examples on the documentation](https://docs.rs/russell_tensor/)
* [Examples directory](https://github.com/cpmech/russell/tree/main/russell_tensor/examples)

### Computing the Invariants

```rust
use russell_tensor::{StrError, Tensor2};

fn main() -> Result<(), StrError> {
    // Allocate a symmetric second-order tensor given the standard components
    let a = Tensor2::<6>::from_std_matrix(&[
        [1.0, 2.0, 3.0],
        [2.0, 2.0, 4.0],
        [3.0, 4.0, 3.0],
    ])?;

    // Compute the principal invariants
    let ii1 = a.invariant_ii1();
    let ii2 = a.invariant_ii2();
    let ii3 = a.invariant_ii3();

    println!("I1 = {:.6}", ii1);
    println!("I2 = {:.6}", ii2);
    println!("I3 = {:.6}", ii3);
    Ok(())
}
```

### Allocating Second Order Tensors

```rust
use russell_tensor::{StrError, Tensor2, SQRT_2};

fn main() -> Result<(), StrError> {
    // Allocate a general second-order tensor given the standard components
    let a = Tensor2::<9>::from_std_matrix(&[
        [1.0, SQRT_2 * 2.0, SQRT_2 * 3.0],
        [SQRT_2 * 4.0, 5.0, SQRT_2 * 6.0],
        [SQRT_2 * 7.0, SQRT_2 * 8.0, 9.0],
    ])?;
    assert_eq!(
        format!("{:.1}", a),
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

    // Allocate a symmetric second-order tensor given the standard components
    let b = Tensor2::<6>::from_std_matrix(&[
        [1.0, 4.0 / SQRT_2, 6.0 / SQRT_2],
        [4.0 / SQRT_2, 2.0, 5.0 / SQRT_2],
        [6.0 / SQRT_2, 5.0 / SQRT_2, 3.0],
    ])?;
    assert_eq!(
        format!("{:.1}", b),
        "┌     ┐\n\
         │ 1.0 │\n\
         │ 2.0 │\n\
         │ 3.0 │\n\
         │ 4.0 │\n\
         │ 5.0 │\n\
         │ 6.0 │\n\
         └     ┘"
    );

    // Allocate a symmetric second-order tensor given the standard components for 2D problems
    let c = Tensor2::<4>::from_std_matrix(
        &[[1.0, 4.0 / SQRT_2, 0.0], [4.0 / SQRT_2, 2.0, 0.0], [0.0, 0.0, 3.0]],
    )?;
    assert_eq!(
        format!("{:.1}", c),
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

## For developers

* This crate depends on `russell_lab`, which requires non-Rust high-performance libraries (see the Installation section)
* Run the examples with `cargo run --example <name>`

