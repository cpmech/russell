# Russell ODE - Solvers for Ordinary Differential Equations and Differential Algebraic Equations

_This crate is part of [Russell - Rust Scientific Library](https://github.com/cpmech/russell)_

## Contents

* [Introduction](#introduction)
* [Installation](#installation)
* [Setting Cargo.toml](#cargo)
* [Examples](#examples)
    * [Simple ODE with a single equation](#simple-single)
    * [Simple system with mass matrix](#simple-mass)
    * [Brusselator ODE](#brusselator-ode)
    * [Hairer-Wanner Equation (1.1)](#hairer-wanner-eq1)
    * [Robertson's Equation](#robertson)
    * [Van der Pol's Equation](#van-der-pol)

## <a name="introduction"></a> Introduction

This library implements (in pure Rust) solvers to ordinary differential equations (ODEs) and differential algebraic equations (DAEs). Specifically, this library implements several explicit Runge-Kutta methods (e.g., Dormand-Prince formulae) and two implicit Runge-Kutta methods, namely the Backward Euler and the Radau IIA of fifth-order (aka Radau5). The Radau5 solver is able to solver DAEs of Index-1, by accepting the so-called *mass matrix*.

The code in this library is based on Hairer-Nørsett-Wanner books and respective Fortran codes (see references [1] and [2]). The code for Dormand-Prince 5(4) and Dormand-Prince 8(5,3) are fairly different than the Fortran counterparts. The code for Radau5 follows closely reference [2]; however some small differences are considered. Despite the coding differences, the numeric results match the Fortran results quite well.

The ODE/DAE system can be easily defined using the System data structure; [see the examples below](#examples).

### References

1. Hairer E, Nørsett, SP, Wanner G (2008) Solving Ordinary Differential Equations I.
   Non-stiff Problems. Second Revised Edition. Corrected 3rd printing 2008. Springer Series
   in Computational Mathematics, 528p
2. Hairer E, Wanner G (2002) Solving Ordinary Differential Equations II.
   Stiff and Differential-Algebraic Problems. Second Revised Edition.
   Corrected 2nd printing 2002. Springer Series in Computational Mathematics, 614p
3. Kreyszig, E (2011) Advanced engineering mathematics; in collaboration with Kreyszig H,
   Edward JN 10th ed 2011, Hoboken, New Jersey, Wiley

## <a name="installation"></a> Installation

This crate depends on `russell_lab`, which, in turn, depends on an efficient BLAS library such as [OpenBLAS](https://github.com/OpenMathLib/OpenBLAS) and [Intel MKL](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2023-2/overview.html).

[The root README file presents the steps to install the required dependencies.](https://github.com/cpmech/russell)

## <a name="cargo"></a> Setting Cargo.toml

[![Crates.io](https://img.shields.io/crates/v/russell_ode.svg)](https://crates.io/crates/russell_ode)

👆 Check the crate version and update your Cargo.toml accordingly:

```toml
[dependencies]
russell_ode = "*"
```

## <a name="examples"></a> Examples

This section illustrates how to use `russell_ode`. More examples:

* [Examples on how to define the ODE/DAE system](https://github.com/cpmech/russell/tree/main/russell_ode/src/samples.rs)
* [Examples directory](https://github.com/cpmech/russell/tree/main/russell_ode/examples)

### <a name="simple-single"></a> Simple ODE with a single equation

Solve the simple ODE with Dormand-Prince 8(5,3):

```text
dy/dx = x + y    with    y(0) = 0
```

See the code [simple_ode_single_equation.rs](https://github.com/cpmech/russell/tree/main/russell_ode/examples/simple_ode_single_equation.rs); reproduced below:

```rust
use russell_lab::{vec_max_abs_diff, StrError, Vector};
use russell_ode::prelude::*;

fn main() -> Result<(), StrError> {
    // ODE system
    let ndim = 1;
    let system = System::new(
        ndim,
        |f, x, y, _args: &mut NoArgs| {
            f[0] = x + y[0];
            Ok(())
        },
        no_jacobian,
        HasJacobian::No,
        None,
        None,
    );

    // solver
    let params = Params::new(Method::DoPri8);
    let mut solver = OdeSolver::new(params, &system)?;

    // initial values
    let x = 0.0;
    let mut y = Vector::from(&[0.0]);

    // solve from x = 0 to x = 1
    let x1 = 1.0;
    let mut args = 0;
    solver.solve(&mut y, x, x1, None, None, &mut args)?;
    println!("y =\n{}", y);

    // check the results
    let y_ana = Vector::from(&[f64::exp(x1) - x1 - 1.0]);
    let (_, error) = vec_max_abs_diff(&y, &y_ana)?;
    println!("error = {:e}", error);
    assert!(error < 1e-8);

    // print stats
    println!("{}", solver.bench());
    Ok(())
}
```

The output looks like:

```text
y =
┌                    ┐
│ 0.7182818250641057 │
└                    ┘
error = 3.39E-09
DoPri8: Dormand-Prince method (explicit, order 8(5,3), embedded)
Number of function evaluations   = 108
Number of performed steps        = 9
Number of accepted steps         = 9
Number of rejected steps         = 0
Last accepted/suggested stepsize = 1.8976857444701694
Max time spent on a step         = 3.789µs
Total time                       = 48.038µs
```

### <a name="simple-mass"></a> Simple system with mass matrix

Solve with Radau5:

```text
y0' + y1'     = -y0 + y1
y0' - y1'     =  y0 + y1
          y2' = 1/(1 + x)

y0(0) = 1,  y1(0) = 0,  y2(0) = 0
```

Thus:

```text
M y' = f(x, y)
```

with:

```text
    ┌          ┐       ┌           ┐
    │  1  1  0 │       │ -y0 + y1  │
M = │  1 -1  0 │   f = │  y0 + y1  │
    │  0  0  1 │       │ 1/(1 + x) │
    └          ┘       └           ┘
```

The Jacobian matrix is:

```text
         ┌          ┐
    df   │ -1  1  0 │
J = —— = │  1  1  0 │
    dy   │  0  0  0 │
         └          ┘
```

The analytical solution is:

```text
y0(x) = cos(x)
y1(x) = -sin(x)
y2(x) = log(1 + x)
```

Reference: [Numerical Solution of Differential-Algebraic Equations: Solving Systems with a Mass Matrix](https://reference.wolfram.com/language/tutorial/NDSolveDAE.html).

See the code [simple_system_with_mass.rs](https://github.com/cpmech/russell/tree/main/russell_ode/examples/simple_system_with_mass.rs); reproduced below:

```rust
use russell_lab::{vec_max_abs_diff, StrError, Vector};
use russell_ode::prelude::*;
use russell_sparse::CooMatrix;

fn main() -> Result<(), StrError> {
    // DAE system
    let ndim = 3;
    let jac_nnz = 4;
    let mut system = System::new(
        ndim,
        |f: &mut Vector, x: f64, y: &Vector, _args: &mut NoArgs| {
            f[0] = -y[0] + y[1];
            f[1] = y[0] + y[1];
            f[2] = 1.0 / (1.0 + x);
            Ok(())
        },
        move |jj: &mut CooMatrix, _x: f64, _y: &Vector, m: f64, _args: &mut NoArgs| {
            jj.reset();
            jj.put(0, 0, m * (-1.0)).unwrap();
            jj.put(0, 1, m * (1.0)).unwrap();
            jj.put(1, 0, m * (1.0)).unwrap();
            jj.put(1, 1, m * (1.0)).unwrap();
            Ok(())
        },
        HasJacobian::Yes,
        Some(jac_nnz),
        None,
    );

    // mass matrix
    let mass_nnz = 5;
    system.init_mass_matrix(mass_nnz).unwrap();
    system.mass_put(0, 0, 1.0).unwrap();
    system.mass_put(0, 1, 1.0).unwrap();
    system.mass_put(1, 0, 1.0).unwrap();
    system.mass_put(1, 1, -1.0).unwrap();
    system.mass_put(2, 2, 1.0).unwrap();

    // solver
    let params = Params::new(Method::Radau5);
    let mut solver = OdeSolver::new(params, &system)?;

    // initial values
    let x = 0.0;
    let mut y = Vector::from(&[1.0, 0.0, 0.0]);

    // solve from x = 0 to x = 20
    let x1 = 20.0;
    let mut args = 0;
    solver.solve(&mut y, x, x1, None, None, &mut args)?;
    println!("y =\n{}", y);

    // check the results
    let y_ana = Vector::from(&[f64::cos(x1), -f64::sin(x1), f64::ln(1.0 + x1)]);
    let (_, error) = vec_max_abs_diff(&y, &y_ana)?;
    println!("error = {:e}", error);
    assert!(error < 1e-4);

    // print stats
    println!("{}", solver.bench());
    Ok(())
}
```

The output looks like:

```text
y =
┌                     ┐
│  0.4081258859665056 │
│ -0.9129961945737365 │
│  3.0445213906613513 │
└                     ┘
error = 5.0943846108819635e-5
Radau5: Radau method (Radau IIA) (implicit, order 5, embedded)
Number of function evaluations   = 204
Number of Jacobian evaluations   = 1
Number of factorizations         = 14
Number of lin sys solutions      = 52
Number of performed steps        = 47
Number of accepted steps         = 47
Number of rejected steps         = 0
Number of iterations (maximum)   = 2
Number of iterations (last step) = 1
Last accepted/suggested stepsize = 0.02811710719458652
Max time spent on a step         = 36.78µs
Max time spent on the Jacobian   = 343ns
Max time spent on factorization  = 8.085901ms
Max time spent on lin solution   = 3.89526ms
Total time                       = 27.107919ms
```

### <a name="brusselator-ode"></a> Brusselator ODE

#### Solving with DoPri8 -- 8(5,3) -- dense output

This is a system of two ODEs, well explained in Reference # 1. This problem is solved with the DoPri8 method (it has a hybrid error estimator of 5th and 3rd order; see Reference # 1).

This example also shows how to enable the *dense output*.

See the code [brusselator_ode_dopri8.rs](https://github.com/cpmech/russell/tree/main/russell_ode/examples/brusselator_ode_dopri8.rs); reproduced below (without the plotting commands):

```rust
use russell_lab::StrError;
use russell_ode::prelude::*;

fn main() -> Result<(), StrError> {
    // get the ODE system
    let (system, mut data, mut args, y_ref) = Samples::brusselator_ode();

    // solver
    let params = Params::new(Method::DoPri8);
    let mut solver = OdeSolver::new(params, &system)?;

    // enable dense output
    let mut out = Output::new();
    let h_out = 0.01;
    let selected_y_components = &[0, 1];
    out.enable_dense(h_out, selected_y_components)?;

    // solve the problem
    solver.solve(&mut data.y0, data.x0, data.x1, None, Some(&mut out), &mut args)?;

    // print the results and stats
    println!("y_russell     = {:?}", data.y0.as_data());
    println!("y_mathematica = {:?}", y_ref.as_data());
    println!("{}", solver.bench());
    Ok(())
}
```

The output looks like this:

```text
y_russell     = [0.4986435155366857, 4.596782273713258]
y_mathematica = [0.49863707126834783, 4.596780349452011]
DoPri8: Dormand-Prince method (explicit, order 8(5,3), embedded)
Number of function evaluations   = 647
Number of performed steps        = 45
Number of accepted steps         = 38
Number of rejected steps         = 7
Last accepted/suggested stepsize = 2.1617616186304227
Max time spent on a step         = 47.643µs
Total time                       = 898.347µs
```

A plot of the (dense) solution is shown below:

![Brusselator results: DoPri8](data/figures/brusselator_dopri8.svg)

#### Variable step sizes

This example solves the Brusselator ODE with variable step sizes for different tolerances. In this example, `tol = abs_tol = rel_tol`.

See the code [brusselator_ode_var_step.rs](https://github.com/cpmech/russell/tree/main/russell_ode/examples/brusselator_ode_var_step.rs)

The results are:

```text
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
       tol =  1.00E-02  1.00E-04  1.00E-06  1.00E-08
      Method     Error     Error     Error     Error
────────────────────────────────────────────────────
      Radau5   1.9E-03   7.9E-06   1.3E-07   3.2E-09
     Merson4   3.8E-02   2.1E-04   9.9E-06   7.1E-08
      DoPri5   8.0E-03   1.7E-04   1.8E-06   2.0E-08
      DoPri8   1.4E-02   6.4E-06   2.7E-07   4.2E-09
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

And the convergence plot is:

![Brusselator results: var step](data/figures/brusselator_ode_var_step.svg)

#### Fixed step sizes

This example solves the Brusselator ODE with fixed step sizes and explicit Runge-Kutta methods.

See the code [brusselator_ode_fix_step.rs](https://github.com/cpmech/russell/tree/main/russell_ode/examples/brusselator_ode_fix_step.rs)

The results are:

```text
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        h = 2.00E-01 1.00E-01 5.00E-02 1.00E-02 1.00E-03
     Method    Error    Error    Error    Error    Error
────────────────────────────────────────────────────────
        Rk2  6.1E-03  4.1E-03  1.5E-03  7.4E-05  7.8E-07
        Rk3  6.1E-03  8.2E-04  1.5E-04  1.7E-06  1.8E-09
      Heun3  1.1E-02  1.3E-03  1.7E-04  1.5E-06  1.5E-09
        Rk4  5.9E-03  2.8E-04  1.7E-05  2.7E-08  2.7E-12
     Rk4alt  7.0E-03  2.4E-04  9.3E-06  1.1E-08  9.9E-13
    MdEuler  4.1E-02  7.2E-03  2.0E-03  9.0E-05  9.2E-07
    Merson4  5.7E-04  6.3E-06  7.9E-07  1.7E-09  1.6E-13
 Zonneveld4  5.9E-03  2.8E-04  1.7E-05  2.7E-08  2.7E-12
  Fehlberg4  9.4E-03  8.1E-05  1.4E-06  1.7E-09  1.7E-13
     DoPri5  1.9E-03  6.1E-05  7.3E-07  4.7E-11  5.7E-14
    Verner6  3.6E-03  3.8E-05  3.5E-07  1.4E-11  4.1E-14
  Fehlberg7  2.5E-05  9.9E-08  6.3E-10  8.9E-15  8.9E-15
     DoPri8  3.9E-06  5.3E-10  5.8E-12  1.7E-14  2.0E-14
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

And the convergence plot is:

![Brusselator results: fix step](data/figures/brusselator_ode_fix_step.svg)

### <a name="hairer-wanner-eq1"></a> Hairer-Wanner Equation (1.1)

This example illustrates the instability of the forward Euler method with step sizes above the stability limit. The equation is (reference # 2, page 2):

```text
dy/dx = -50 (y - cos(x))          (1.1)
```

This example also shows how to enable the output of accepted steps.

See the code [hairer_wanner_eq1.rs](https://github.com/cpmech/russell/tree/main/russell_ode/examples/hairer_wanner_eq1.rs)

The results are show below:

![Hairer-Wanner Eq(1.1)](data/figures/hairer_wanner_eq1.svg)

### <a name="robertson"></a> Robertson's Equation

This example illustrates the Robertson's equation. In this problem DoPri5 uses many steps (about 200). On the other hand, Radau5 solves the problem with 17 accepted steps.

This example also shows how to enable the output of accepted steps.

See the code [robertson.rs](https://github.com/cpmech/russell/tree/main/russell_ode/examples/robertson.rs)

The solution obtained with Radau5 and DoPri5 using two sets of tolerances are illustrated below:

![Robertson's Equation - Solution](data/figures/robertson_a.svg)

The step sizes from the DoPri solution with Tol = 1e-2 are illustrated below:

![Robertson's Equation - Step Sizes](data/figures/robertson_b.svg)

### <a name="van-der-pol"></a> Van der Pol's Equation

This example illustrated the *stiffness* of the Van der Pol problem (equation + initial conditions + step size + method). In this example, DoPri5 with Tol = 1e-3 is used.

This example also shows how to enable the stiffness detection.

See the code [van_der_pol_dopri5.rs](https://github.com/cpmech/russell/tree/main/russell_ode/examples/van_der_pol_dopri5.rs)

The results are show below:

![Van der Pol's Equation - DoPri5](data/figures/van_der_pol_dopri5.svg)

The figure's red dashed lines mark the moment when stiffness has been detected first. The stiffness is confirmed after 15 accepted steps with repeated stiffness thresholds being reached. The positive thresholds are counted when h·ρ becomes greater than the corresponding factor·max(h·ρ)---the value on the stability limit (3.3 for DoPri5; factor ~= 0.976). Note that ρ is the approximation of the dominant eigenvalue of the Jacobian. After 6 accepted steps, if the thresholds are not reached, the stiffness detection flag is set to false.
