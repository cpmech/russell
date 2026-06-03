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
    - [Example 1: Solving 1D Poisson equation with Finite Differences](#example-1-solving-1d-poisson-equation-with-finite-differences)
    - [Example 2: Solving 1D problems with Spectral Collocation](#example-2-solving-1d-problems-with-spectral-collocation)
    - [Example 3: Solving 2D Poisson equation](#example-3-solving-2d-poisson-equation)
    - [Example 4: Using Lagrange multipliers method](#example-4-using-lagrange-multipliers-method)



## Introduction

This library implements essential tools to solve partial differential equations (PDEs). It does not implement full-fledge PDE solvers for general problems and, hence, this library is quite limited.

A goal is to provide tools to test other crates such as `russell_ode` and `russell_nonlinear` because they employ PDE problems as **testing** platforms.

Currently, simple finite differences operators are implemented, in addition to spectral collocation methods in 1D and 2D. The library also implements the transfinite mapping method to generate meshes on non-rectangular domains.

The linear systems are solved using the System Partitioning Strategy (SPS) or the Lagrange Multipliers Method (LMM).

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
* `local_sparse`: Use locally compiled SuiteSparse and MUMPS

Note that the [main README file](https://github.com/cpmech/russell) presents the steps to compile the required libraries according to each feature.




## 🌟 Examples

This section illustrates how to use `russell_pde`. See also:

* [More examples on the documentation](https://docs.rs/russell_pde/)
* [Examples directory](https://github.com/cpmech/russell/tree/main/russell_pde/examples)



### Example 1: Solving 1D Poisson equation with Finite Differences

This example solves the Poisson equation in 1D using the Finite Difference Method (FDM):

```text
  ∂²ϕ
- ——— = x    on [0, 1]
  ∂x²

With boundary conditions: ϕ(0) = 0, ϕ(1) = 0
```

The analytical solution is: `ϕ(x) = (x - x³) / 6`

```rust
use russell_lab::approx_eq;
use russell_pde::{Fdm1d, Grid1d, EssentialBcs1d, NaturalBcs1d, StrError};

fn main() -> Result<(), StrError> {
    // Define the problem domain and diffusion coefficient
    let (xmin, xmax) = (0.0, 1.0);
    let kx = 1.0;

    // Set up boundary conditions
    let mut ebcs = EssentialBcs1d::new();
    ebcs.set_homogeneous(); // ϕ(0) = 0, ϕ(1) = 0

    let nbcs = NaturalBcs1d::new();

    // Create uniform grid with 10 subdivisions
    let nx = 10;
    let grid = Grid1d::new_uniform(xmin, xmax, nx)?;

    // Create the solver
    let mut fdm = Fdm1d::new(grid, ebcs, nbcs, kx)?;

    // Define the source term f(x) = x
    let source = |x: f64| x;

    // Solve using System Partitioning Strategy
    let solution = fdm.solve_sps(0.0, source)?;

    // Verify against analytical solution
    let analytical = |x: f64| (x - x.powi(3)) / 6.0;
    fdm.for_each_coord(|m, x| {
        approx_eq(solution[m], analytical(x), 1e-3);
        println!("x = {:.3}, ϕ = {:.6}", x, solution[m]);
    });

    Ok(())
}
```

### Example 2: Solving 1D problems with Spectral Collocation

For higher accuracy, use the Spectral Collocation method instead:

```rust
use russell_pde::{Spc1d, EssentialBcs1d, NaturalBcs1d, StrError};

fn main() -> Result<(), StrError> {
    let (xmin, xmax) = (0.0, 1.0);
    let kx = 1.0;

    // Set up boundary conditions
    let mut ebcs = EssentialBcs1d::new();
    ebcs.set_homogeneous();
    let nbcs = NaturalBcs1d::new();

    // Create spectral collocation solver with N=8 polynomial degree
    let nx = 8;
    let mut spc = Spc1d::new(xmin, xmax, nx, ebcs, nbcs, kx)?;

    // Solve the problem
    let source = |x: f64| x;
    let solution = spc.solve_sps(0.0, source)?;

    // Calculate flow vectors (derivative information)
    let flow = spc.calculate_flow_vectors(&solution)?;

    println!("Solution computed with spectral accuracy!");
    Ok(())
}
```

### Example 3: Solving 2D Poisson equation

Solve the 2D Poisson equation on a rectangular domain:

```text
  ∂²ϕ    ∂²ϕ
- ——— -  ——— = f(x,y)    on [0,1] × [0,1]
  ∂x²    ∂y²
```

```rust
use russell_pde::{Fdm2d, Grid2d, EssentialBcs2d, NaturalBcs2d, Side, StrError};
use russell_lab::math::PI;

fn main() -> Result<(), StrError> {
    // Define 2D domain
    let (xmin, xmax) = (0.0, 1.0);
    let (ymin, ymax) = (0.0, 1.0);
    let (kx, ky) = (1.0, 1.0);

    // Set boundary conditions
    let mut ebcs = EssentialBcs2d::new();
    ebcs.set(Side::Xmin, |_, _| 0.0);
    ebcs.set(Side::Xmax, |_, _| 0.0);
    ebcs.set(Side::Ymin, |x, _| (PI * x).sin());
    ebcs.set(Side::Ymax, |_, _| 0.0);

    let nbcs = NaturalBcs2d::new();

    // Create uniform grid
    let (nx, ny) = (20, 20);
    let grid = Grid2d::new_uniform(xmin, xmax, ymin, ymax, nx, ny)?;

    // Create solver and solve
    let fdm = Fdm2d::new(grid, ebcs, nbcs, kx, ky)?;
    let source = |_x: f64, _y: f64| 0.0;
    let solution = fdm.solve_sps(0.0, &source)?;

    println!("2D solution computed on {}×{} grid", nx, ny);
    Ok(())
}
```

### Example 4: Using Lagrange multipliers method

Both SPS (System Partitioning Strategy) and LMM (Lagrange Multipliers Method) are available:

```rust
use russell_pde::{Fdm1d, Grid1d, EssentialBcs1d, NaturalBcs1d, StrError};

fn main() -> Result<(), StrError> {
    let (xmin, xmax) = (0.0, 1.0);
    let kx = 1.0;
    let mut ebcs = EssentialBcs1d::new();
    ebcs.set_homogeneous();
    let nbcs = NaturalBcs1d::new();

    let grid = Grid1d::new_uniform(xmin, xmax, 10)?;
    let mut fdm = Fdm1d::new(grid, ebcs, nbcs, kx)?;

    let source = |x: f64| x;

    // Method 1: System Partitioning Strategy (default)
    let solution_sps = fdm.solve_sps(0.0, source)?;

    // Method 2: Lagrange Multipliers Method
    let solution_lmm = fdm.solve_lmm(0.0, source)?;

    // Both methods produce the same solution
    Ok(())
}
```

See the [examples directory](https://github.com/cpmech/russell/tree/main/russell_pde/examples) for more advanced usage including mapped domains and transfinite interpolation.
