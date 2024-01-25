// use crate::Information;
use russell_lab::Vector;

pub(crate) trait OdeSolverTrait<A> {
    /// Calculates the quantities required to update x0 and y0
    ///
    /// Returns the (`relative_error`, `stiffness_ratio`)
    fn step(&mut self, x0: f64, y0: &Vector, h: f64, first_step: bool, args: &mut A) -> (f64, f64);

    /// Accepts the update and computes the next stepsize
    ///
    /// Returns `stepsize_new`
    fn accept(
        &mut self,
        y0: &mut Vector,
        x0: f64,
        h: f64,
        relative_error: f64,
        previous_relative_error: f64,
        args: &mut A,
    ) -> f64;

    /// Rejects the update
    ///
    /// Returns `stepsize_new`
    fn reject(&mut self, h: f64, relative_error: f64) -> f64;

    /// Computes the dense output
    fn dense_output(&self, yout: &mut Vector, h: f64, x: f64, y: &Vector, xout: f64);
}
