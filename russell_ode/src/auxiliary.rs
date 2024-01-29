use crate::StrError;
use russell_lab::Vector;
use russell_sparse::CooMatrix;

/// Returns an error to indicate that the Jacobian function is not available
///
/// **Note:** Use this function with [HasJacobian::No]
pub fn no_jacobian<A>(
    _jj: &mut CooMatrix,
    _x: f64,
    _y: &Vector,
    _multiplier: f64,
    _args: &mut A,
) -> Result<(), StrError> {
    Err("analytical Jacobian is not available")
}

/// Disables the output of accepted steps
pub fn no_step_output(_step: usize, _h: f64, _x: f64, _y: &Vector) -> Result<bool, StrError> {
    Ok(false)
}

/// Disables the dense output
pub fn no_dense_output(
    _y_out: &mut Vector,
    _x_out: f64,
    _step: usize,
    _h: f64,
    _x: f64,
    _y: &Vector,
) -> Result<bool, StrError> {
    Ok(false)
}
