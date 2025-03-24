# Russell Nonlin - Numerical Continuation methods to solve nonlinear systems of equations

[![documentation](https://docs.rs/russell_nonlin/badge.svg)](https://docs.rs/russell_nonlin/)

_This crate is part of [Russell - Rust Scientific Library](https://github.com/cpmech/russell)_

## Contents <!-- omit from toc --> 

- [Russell Nonlin - Numerical Continuation methods to solve nonlinear systems of equations](#russell-nonlin---numerical-continuation-methods-to-solve-nonlinear-systems-of-equations)
  - [Introduction](#introduction)
    - [Documentation](#documentation)
    - [References](#references)
  - [Installation](#installation)
    - [Setting Cargo.toml](#setting-cargotoml)
    - [Optional features](#optional-features)
  - [🌟 Examples](#-examples)
    - [Simple example](#simple-example)



## Introduction

This library implements solvers for nonlinear systems of equations using *Numerical Continuation*, in particular, Predictor-Corrector algorithms based on the Euler-Newton(-Raphson) method (See References #1 and #2).

### Documentation

* [![documentation](https://docs.rs/russell_nonlin/badge.svg)](https://docs.rs/russell_nonlin/) — [russell_nonlin documentation](https://docs.rs/russell_nonlin/)

### References

1. Doedel EJ (2007) Lecture Notes on Numerical Analysis of Nonlinear Equations. Numerical Continuation Methods for Dynamical Systems: Path following and boundary value problems. Ed. by B Krauskopf, HM Osinga, J Galán-Vioque. Dordrecht: Springer Netherlands, pp. 1–49. doi: 10.1007/978-1-4020-6356-5 1
2. Allgower EL, Georg K (2003) Introduction to Numerical Continuation Methods. Society for Industrial and Applied Mathematics. doi: 10.1137/ 1.9780898719154



## Installation

This crate depends on some non-rust high-performance libraries. [See the main README file for the steps to install these dependencies.](https://github.com/cpmech/russell)



### Setting Cargo.toml

[![Crates.io](https://img.shields.io/crates/v/russell_nonlin.svg)](https://crates.io/crates/russell_nonlin)

👆 Check the crate version and update your Cargo.toml accordingly:

```toml
[dependencies]
russell_nonlin = "*"
```

### Optional features

The following (Rust) features are available:

* `intel_mkl`: Use Intel MKL instead of OpenBLAS
* `local_suitesparse`: Use a locally compiled version of SuiteSparse
* `with_mumps`: Enable the MUMPS solver (locally compiled)

Note that the [main README file](https://github.com/cpmech/russell) presents the steps to compile the required libraries according to each feature.




## 🌟 Examples

This section illustrates how to use `russell_nonlin`. See also:

* [More examples on the documentation](https://docs.rs/russell_nonlin/)
* [Examples directory](https://github.com/cpmech/russell/tree/main/russell_nonlin/examples)



### Simple example

TODO
