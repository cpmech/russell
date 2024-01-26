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
/// # Returns
///
/// * `ok` -- a flag to indicate that the simulation is OK
///     * `ok = true` means that the simulation should continue
///     * `ok = false` tells the solver to stop, nicely
pub type OutputStep = fn(step: usize, h: f64, x: f64, y: &Vector) -> Result<bool, StrError>;
// pub type OutputStep<B> = fn(step: usize, h: f64, x: f64, y: &Vector, out_args: &mut B) -> Result<bool, StrError>;

pub fn out_step_none(_step: usize, _h: f64, _x: f64, _y: &Vector, _out_args: &mut u8) -> Result<bool, StrError> {
    Ok(true)
}

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
/// # Returns
///
/// * `ok` -- a flag to indicate that the simulation is OK
///     * `ok = true` means that the simulation should continue
///     * `ok = false` tells the solver to stop, nicely
pub type OutputDense =
    fn(step: usize, h: f64, x: f64, y: &Vector, x_out: f64, y_out: &Vector) -> Result<bool, StrError>;
// pub type OutputDense<C> = fn(
//     step: usize,
//     h: f64,
//     x: f64,
//     y: &Vector,
//     x_out: f64,
//     y_out: &Vector,
//     dense_args: &mut C,
// ) -> Result<bool, StrError>;

pub fn out_dense_none(
    _step: usize,
    _h: f64,
    _x: f64,
    _y: &Vector,
    _x_out: f64,
    _y_out: &Vector,
    _dense_args: &mut u8,
) -> Result<bool, StrError> {
    Ok(true)
}
