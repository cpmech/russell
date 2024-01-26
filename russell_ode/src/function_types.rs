#![allow(unused)]

use crate::StrError;
use russell_lab::Vector;
use russell_sparse::SparseMatrix;

/// Defines the ODE system
///
/// ```text
/// d{y}
/// ———— = {f}(x, {y})
///  dx
/// ```
///
/// Note: the function receives the stepsize h as well.
///
/// # Input
///
/// * `f` -- (output) the function: `{f}(x, {y})`
/// * `x` -- current x
/// * `y` -- current {y}
pub type OdeSys<A> = fn(f: &mut Vector, x: f64, y: &Vector, args: &mut A) -> Result<(), StrError>;

/// Defines the Jacobian matrix of the ODE system
///
/// ```text
/// d{f}
/// ———— = [J](x, {y})
/// d{y}
/// ```
///
/// Note: the Jacobian function receives the stepsize h as well.
///
/// # Input
///
/// * `jj` -- (output) Jacobian matrix `d{f}/d{y} := [J](x, {y})`
/// * `x` -- current x
/// * `y` -- current {y}
pub type OdeSysJac<A> = fn(jj: &mut SparseMatrix, x: f64, y: &Vector, args: &mut A) -> Result<(), StrError>;

/// Defines a function to be called when a step is accepted
///
/// # Input
///
/// * `step` -- index of step (0 is the very first output whereas 1 is the first accepted step)
/// * `h` -- stepsize = dx
/// * `x` -- scalar variable
/// * `y` -- vector variable
///
/// # Output
///
/// * `stop` -- flag to stop the simulation (nicely)
pub type OutputStep = fn(step: usize, h: f64, x: f64, y: &Vector) -> Result<bool, StrError>;

/// Defines a function to generate the dense output
///
/// The dense output is generated for (many) equally spaced points, regardless of the actual stepsize.
///
/// # Input
///
/// * `step` -- index of step (0 is the very first output whereas 1 is the first accepted step)
/// * `h` -- current (optimal) step size
/// * `x` -- current (just updated) x
/// * `y` -- current (just updated) y
/// * `x_out` -- selected x to produce an output
/// * `y_out` -- y values computed @ x_out
///
/// # Output
///
/// * `stop` -- flag to stop the simulation (nicely)
pub type OutputDense =
    fn(step: usize, h: f64, x: f64, y: &Vector, x_out: f64, y_out: &Vector) -> Result<bool, StrError>;
