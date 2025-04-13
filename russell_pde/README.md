# Russell PDE - Essential tools to solve partial differential equations; not a full-fledged PDE solver

[![documentation](https://docs.rs/russell_pde/badge.svg)](https://docs.rs/russell_pde/)

_This crate is part of [Russell - Rust Scientific Library](https://github.com/cpmech/russell)_

## Contents <!-- omit from toc --> 

- [Russell PDE - Essential tools to solve partial differential equations; not a full-fledged PDE solver](#russell-pde---essential-tools-to-solve-partial-differential-equations-not-a-full-fledged-pde-solver)
  - [Introduction](#introduction)
    - [Documentation](#documentation)
  - [Installation](#installation)
    - [Setting Cargo.toml](#setting-cargotoml)
    - [Optional features](#optional-features)
  - [🌟 Examples](#-examples)
    - [Simple example](#simple-example)



## Introduction

This library implements essential tools to solve partial differential equations (PDEs). It does not implement full-fledge PDE solvers for general problems and, hence, this library is quite limited.

Currently, a simple finite differences Lapoperator

### Documentation

* [![documentation](https://docs.rs/russell_pde/badge.svg)](https://docs.rs/russell_pde/) — [russell_pde documentation](https://docs.rs/russell_pde/)




## Installation

This crate depends on some non-rust high-performance libraries. [See the main README file for the steps to install these dependencies.](https://github.com/cpmech/russell)



### Setting Cargo.toml

[![Crates.io](https://img.shields.io/crates/v/russell_pde.svg)](https://crates.io/crates/russell_pde)

👆 Check the crate version and update your Cargo.toml accordingly:

```toml
[dependencies]
russell_pde = "*"
```

### Optional features

The following (Rust) features are available:

* `intel_mkl`: Use Intel MKL instead of OpenBLAS
* `local_suitesparse`: Use a locally compiled version of SuiteSparse
* `with_mumps`: Enable the MUMPS solver (locally compiled)

Note that the [main README file](https://github.com/cpmech/russell) presents the steps to compile the required libraries according to each feature.




## 🌟 Examples

This section illustrates how to use `russell_pde`. See also:

* [More examples on the documentation](https://docs.rs/russell_pde/)
* [Examples directory](https://github.com/cpmech/russell/tree/main/russell_pde/examples)



### Simple example

TODO
