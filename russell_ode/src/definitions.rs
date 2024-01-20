use crate::StrError;
use russell_lab::Vector;
use russell_sparse::SparseMatrix;

/// Defines the main function d{y}/dx = {f}(x, {y})
///
/// ```text
/// d{y}
/// ———— = {f}(h=dx, x, {y})
///  dx
/// ```
///
/// Note: the function receives the stepsize h as well.
///
/// # Input
///
/// * `f` -- (output) the function: `{f}(h, x, {y})`
/// * `h` -- current stepsize = dx
/// * `x` -- current x
/// * `y` -- current {y}
pub type Func = fn(f: &mut Vector, h: f64, x: f64, y: &Vector) -> Result<(), StrError>;

/// Defines the Jacobian matrix of Func
///
/// ```text
/// d{f}
/// ———— = [J](h=dx, x, {y})
/// d{y}
/// ```
///
/// Note: the Jacobian function receives the stepsize h as well.
///
/// # Input
///
/// * `df_dy` -- (output) Jacobian matrix `d{f}/d{y} := [J](h=dx, x, {y})`
/// * `h` -- current stepsize = dx
/// * `x` -- current x
/// * `y` -- current {y}
pub type JacF = fn(df_dy: &mut SparseMatrix, h: f64, x: f64, y: &Vector) -> Result<(), StrError>;

/// Defines a callback function to be called when a step is accepted
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
pub type StepOutF = fn(step: usize, h: f64, x: f64, y: &Vector) -> Result<bool, StrError>;

/// Defines a function to produce a dense output
///
/// The dense output is produced for (many) equally spaced points, regardless of the actual stepsize.
///
/// # Input
///
/// * `step` -- index of step (0 is the very first output whereas 1 is the first accepted step)
/// * `h` -- best (current) h
/// * `x` -- current (just updated) x
/// * `y` -- current (just updated) y
/// * `x_out` -- selected x to produce an output
/// * `y_out` -- y values computed @ x_out
///
/// # Output
///
/// * `stop` -- flag to stop the simulation (nicely)
pub type DenseOutF = fn(step: usize, h: f64, x: f64, y: &Vector, x_out: f64, y_out: &Vector) -> Result<bool, StrError>;

/// YanaF defines a function to be used when computing analytical solutions
///
/// # Input
///
pub type YanaF = fn(res: &[f64], x: f64) -> Result<(), StrError>;
