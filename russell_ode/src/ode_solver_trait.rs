// use crate::Information;
use russell_lab::Vector;

pub(crate) trait OdeSolverTrait<A> {
    /// Gathers information about the method
    // fn information(&self) -> Information;

    /// Initializes the solver
    // fn initialize(&mut self);

    /// Performs the next step
    ///
    /// Returns the (`relative_error`, `sitffness_ratio`)
    fn next_step(&mut self, xa: f64, ya: &Vector, h: f64, first: bool, args: &mut A) -> (f64, f64);

    /// Accepts the update and computes the next stepsize
    ///
    /// Returns `stepsize_new`
    fn accept_update(
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
    fn reject_update(&mut self, h: f64, relative_error: f64) -> f64;

    /// Computes the dense output
    fn dense_output(&self, yout: &mut Vector, h: f64, x: f64, y: &Vector, xout: f64);
}
