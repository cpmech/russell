# Russell Tensor - Tensor analysis, calculus, and functions for continuum mechanics <!-- omit from toc --> 

[![documentation](https://docs.rs/russell_tensor/badge.svg)](https://docs.rs/russell_tensor/)

_This crate is part of [Russell - Rust Scientific Library](https://github.com/cpmech/russell)_

## Contents <!-- omit from toc --> 

- [Introduction](#introduction)
  - [Capabilities](#capabilities)
  - [Kelvin notation](#kelvin-notation)
  - [Documentation](#documentation)
- [Installation](#installation)
  - [Setting Cargo.toml](#setting-cargotoml)
  - [Optional features](#optional-features)
- [🌟 Examples](#-examples)
  - [Computing the Invariants](#computing-the-invariants)
  - [Allocating Second Order Tensors](#allocating-second-order-tensors)
- [For developers](#for-developers)



## Introduction

This library implements structures and functions for tensor analysis and calculus, with focus on applications in engineering and [Continuum Mechanics](https://en.wikipedia.org/wiki/Continuum_mechanics). The essential functionality for the targeted applications includes second-order and fourth-order tensors, scalar "invariants," and derivatives.

### Capabilities

* `Tensor2` — second-order tensors (symmetric or not) with functions such as the determinant, inverse, norm, and invariants (principal, deviatoric, Lode, octahedral, ...)
* `Tensor4` — fourth-order tensors (minor-symmetric or not)
* Operations between tensors — addition, single and double contractions (dot and ddot), and dyadic products
* Analytical derivatives — first and second derivatives of invariants and tensor functions (e.g., the inverse and squared tensors) with respect to tensors
* `Spectral2` — the spectral (eigen) representation of symmetric second-order tensors
* `LinElasticity` — the linear elasticity equations for small-strain problems (Hooke's law)
* Constants — identity, transposition, and projector tensors

### Kelvin notation

Internally, tensors are stored in the Kelvin basis (Kelvin notation), an isometric (norm-preserving) alternative to [Voigt notation](https://en.wikipedia.org/wiki/Voigt_notation).

In the Kelvin basis, a second-order tensor is mapped to a column matrix (vector) and a fourth-order tensor is mapped to a square matrix. The `√2` factors make the mapping isometric; thus the tensor norm is preserved and standard matrix/vector operations can be used directly.

The `Rep` enum specifies the available representations:

* `Rep::General` — 9×1 / 9×9 (all components)
* `Rep::Symmetric` — 6×1 / 6×6 (symmetric tensors in 3D)
* `Rep::Symmetric2D` — 4×1 / 4×4 (symmetric tensors in 2D)

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

Note that the [main README file](https://github.com/cpmech/russell) presents the steps to compile the required libraries according to each feature.



## 🌟 Examples

This section illustrates how to use `russell_tensor`. See also:

* [More examples on the documentation](https://docs.rs/russell_tensor/)
* [Examples directory](https://github.com/cpmech/russell/tree/main/russell_tensor/examples)

### Computing the Invariants

```rust
use russell_tensor::{Rep, StrError, Tensor2};

fn main() -> Result<(), StrError> {
    // Allocate a symmetric second-order tensor given the standard components
    let sigma = Tensor2::from_std_matrix(
        &[
            [1.0, 2.0, 3.0],
            [2.0, 2.0, 4.0],
            [3.0, 4.0, 3.0],
        ],
        Rep::Symmetric,
    )?;

    // Compute the principal invariants
    let ii1 = sigma.invariant_ii1();
    let ii2 = sigma.invariant_ii2();
    let ii3 = sigma.invariant_ii3();

    println!("I1 = {:.6}", ii1);
    println!("I2 = {:.6}", ii2);
    println!("I3 = {:.6}", ii3);
    Ok(())
}
```

### Allocating Second Order Tensors

```rust
use russell_tensor::{Rep, StrError, Tensor2, SQRT_2};

fn main() -> Result<(), StrError> {
    // Allocate a general second-order tensor given the standard components
    let a = Tensor2::from_std_matrix(
        &[
            [1.0, SQRT_2 * 2.0, SQRT_2 * 3.0],
            [SQRT_2 * 4.0, 5.0, SQRT_2 * 6.0],
            [SQRT_2 * 7.0, SQRT_2 * 8.0, 9.0],
        ],
        Rep::General,
    )?;
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
    let b = Tensor2::from_std_matrix(
        &[
            [1.0, 4.0 / SQRT_2, 6.0 / SQRT_2],
            [4.0 / SQRT_2, 2.0, 5.0 / SQRT_2],
            [6.0 / SQRT_2, 5.0 / SQRT_2, 3.0],
        ],
        Rep::Symmetric,
    )?;
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
    let c = Tensor2::from_std_matrix(
        &[[1.0, 4.0 / SQRT_2, 0.0], [4.0 / SQRT_2, 2.0, 0.0], [0.0, 0.0, 3.0]],
        Rep::Symmetric2D,
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

* This crate is pure Rust with no C dependencies
* Run the examples with `cargo run --example <name>`
