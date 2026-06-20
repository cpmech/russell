# Russell Tensor - Tensor analysis, calculus, and functions for continuum mechanics <!-- omit from toc --> 

[![documentation](https://docs.rs/russell_tensor/badge.svg)](https://docs.rs/russell_tensor/)

_This crate is part of [Russell - Rust Scientific Library](https://github.com/cpmech/russell)_

## Contents <!-- omit from toc --> 

- [Introduction](#introduction)
  - [Documentation](#documentation)
- [Installation](#installation)
  - [Setting Cargo.toml](#setting-cargotoml)
  - [Optional features](#optional-features)
- [🌟 Examples](#-examples)
  - [Allocating Second Order Tensors](#allocating-second-order-tensors)
- [For developers](#for-developers)



## Introduction

This library implements structures and functions for tensor analysis and calculus. The library focuses on applications in engineering and [Continuum Mechanics](https://en.wikipedia.org/wiki/Continuum_mechanics). The essential functionality for the targeted applications includes second-order and fourth-order tensors, scalar "invariants," and derivatives.

This library implements derivatives for scalar functions with respect to tensors, tensor functions with respect to tensors, and others. A convenient basis representation known as Mandel basis (similar to Voigt notation) is considered by this library internally. The user may also use the Mandel basis to perform simpler matrix-vector operations directly.

### Documentation

* [![documentation](https://docs.rs/russell_tensor/badge.svg)](https://docs.rs/russell_tensor/) — [russell_tensor documentation](https://docs.rs/russell_tensor/)



## Installation

This crate depends on some non-rust high-performance libraries. [See the main README file for the steps to install these dependencies.](https://github.com/cpmech/russell)



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

### Allocating Second Order Tensors

```rust
use russell_tensor::{Mandel, StrError, Tensor2, SQRT_2};

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
        format!("{:.1}", a.vector()),
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
        format!("{:.1}", b.vector()),
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
        format!("{:.1}", c.vector()),
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
