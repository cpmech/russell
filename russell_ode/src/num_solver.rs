use crate::{BenchInfo, StrError};
use russell_lab::Vector;

/// Defines the numerical solver
pub(crate) trait NumSolver<A> {
    /// Returns an access to the benchmark object
    fn bench(&mut self) -> &mut BenchInfo;

    /// Initializes the internal variables
    fn initialize(&mut self, x: f64, y: &Vector);

    /// Calculates the quantities required to update x and y
    ///
    /// Returns the (`relative_error`, `stiffness_ratio`)
    fn step(&mut self, x: f64, y: &Vector, h: f64, args: &mut A) -> Result<(f64, f64), StrError>;

    /// Accepts the update and computes the next stepsize
    ///
    /// Returns `stepsize_new`
    fn accept(
        &mut self,
        y: &mut Vector,
        x: f64,
        h: f64,
        relative_error: f64,
        previous_relative_error: f64,
        args: &mut A,
    ) -> Result<f64, StrError>;

    /// Rejects the update
    ///
    /// Returns `stepsize_new`
    fn reject(&mut self, h: f64, relative_error: f64) -> f64;

    /// Computes the dense output
    fn dense_output(&self, y_out: &mut Vector, h: f64, x: f64, x_out: f64);
}
