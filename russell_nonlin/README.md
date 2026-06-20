# Russell Nonlin - Numerical Continuation methods to solve nonlinear systems of equations <!-- omit from toc -->

[![documentation](https://docs.rs/russell_nonlin/badge.svg)](https://docs.rs/russell_nonlin/)

_This crate is part of [Russell - Rust Scientific Library](https://github.com/cpmech/russell)_

## Contents <!-- omit from toc --> 

- [Introduction](#introduction)
  - [Documentation](#documentation)
  - [References](#references)
- [Installation](#installation)
  - [Setting Cargo.toml](#setting-cargotoml)
  - [Optional features](#optional-features)
- [🌟 Examples](#-examples)
  - [Simple example](#simple-example)
  - [Tracing a B-spline curve](#tracing-a-b-spline-curve)
- [For developers](#for-developers)



## Introduction

This library implements solvers for nonlinear systems of equations using *Numerical Continuation*, in particular, Predictor-Corrector algorithms based on the Euler-Newton(-Raphson) method (See References #1 and #2).

Two continuation methods are available:

- **Natural parameter continuation** (`Method::Natural`): treats λ as the explicit control parameter. Solves `G(u, λ) = 0` step-by-step by incrementing λ. Simple and reliable on regular branches, but cannot follow folds (limit points) in the solution curve.
- **Pseudo-arclength continuation** (`Method::Arclength`): parametrizes the solution curve by an arclength-like variable `s`. Solves `G(u(s), λ(s)) = 0` and can navigate around folds, turning points, and other singularities.

The typical workflow is:

1. Define the nonlinear system with `System::new`, providing callbacks for `G(u, λ)` and its Jacobian `∂G/∂u`.
2. Configure the solver with `Config`, choosing the continuation method and tolerances.
3. Create a `Solver` and, optionally, an `Output` to record accepted steps or trigger a callback.
4. Call `Solver::solve` with the initial state `(u₀, λ₀)`, a direction (`IniDir`), a `Stop` criterion, and a `DeltaLambda` step strategy.

### Documentation

* [![documentation](https://docs.rs/russell_nonlin/badge.svg)](https://docs.rs/russell_nonlin/) — [russell_nonlin documentation](https://docs.rs/russell_nonlin/)

### References

1. Doedel EJ (2007) Lecture Notes on Numerical Analysis of Nonlinear Equations. In Numerical Continuation Methods for Dynamical Systems: Path following and boundary value problems. Ed. by B Krauskopf, HM Osinga, J Galán-Vioque. Springer Netherlands, doi: 10.1007/978-1-4020-6356-5 1
2. Mittelmann HD, Fischer B, Eds. (1990) Continuation Techniques and Bifurcation Problems. International Series of Numerical Mathematics, doi: 10.1007/978-3-0348-5681-2



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
* `local_sparse`: Use locally compiled SuiteSparse and MUMPS
* `cudss`: Enable the NVIDIA cuDSS GPU solver

Note that the [main README file](https://github.com/cpmech/russell) presents the steps to compile the required libraries according to each feature.




## 🌟 Examples

This section illustrates how to use `russell_nonlin`. See also:

* [More examples on the documentation](https://docs.rs/russell_nonlin/)
* [Integration tests directory](https://github.com/cpmech/russell/tree/main/russell_nonlin/tests)



### Simple example

This example uses the built-in `Samples::simple_linear_problem` (which defines `G(u, λ) = u - λ`)
and traces the solution branch from λ = 0 to λ = 1 with the Natural parameter continuation method.
The exact solution is `u = λ` throughout.

```rust
use russell_nonlin::{Config, DeltaLambda, IniDir, Output, Samples, Solver, Stop};
use russell_sparse::Sym;

// G(u, λ) = u - λ  →  exact solution: u = λ
let (system, mut u, mut l, mut args) = Samples::simple_linear_problem(Sym::No);

// configure the solver (Natural parameter continuation is the default method)
let config = Config::new();
let mut solver = Solver::new(&config, system).unwrap();

// record λ and u[0] at each accepted step
let out = &mut Output::new();
out.set_recording(true, &[0], &[]);

// trace the solution from λ = 0 to λ = 1 using constant steps of Δλ = 0.1
solver
    .solve(
        &mut args,
        &mut u,
        &mut l,
        IniDir::Pos,
        Stop::MaxLambda(1.0),
        &DeltaLambda::constant(0.1),
        Some(out),
    )
    .unwrap();

// the solution path follows u = λ exactly
assert_eq!(out.get_l_values().len(), 11); // initial point + 10 steps
assert!((u[0] - 1.0).abs() < 1e-14);
assert!((l - 1.0).abs() < 1e-14);
```

### Tracing a B-spline curve

This example traces a B-spline curve defined by `G(u, λ) = 0` using the Pseudo-arclength continuation method. The solution points are plotted together with the B-spline curve.

The nonlinear problem is defined as follows:

```text
G(u, λ) = u - C(λ)
```

where C(λ) is a point on a 2D B-spline curve parametrized by λ ∈ `[0,1]`.

[See the code](https://github.com/cpmech/russell/tree/main/russell_nonlin/examples/arclength_bspline.rs)

The output looks like this:

```text
Pseudo-arclength continuation; solves G(u(s), λ(s)) = 0 (auto)
Using numerical Jacobian         = false
Number of function evaluations   = 223
Number of Jacobian evaluations   = 176
Number of factorizations         = 176
Number of lin sys solutions      = 258
Number of accepted steps         = 82
Number of rejected steps         = 0
Number of performed steps        = 82
Number of iterations (total)     = 223
Last accepted/suggested stepsize = 0.0017592577896469867
```

And the plot looks like this:

![B-spline curve](data/figures/doc_arclength_bspline.svg)

## For developers

* This crate is pure Rust with no C dependencies
* Run the examples with `cargo run --example <name>`
