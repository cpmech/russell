// use crate::Information;
use russell_lab::Vector;

pub(crate) trait OdeSolverTrait<A> {
    /// Gathers information about the method
    // fn information(&self) -> Information;

    /// Initializes the solver
    // fn initialize(&mut self);

    /// Performs the next step
    fn next_step(&mut self, xa: f64, ya: &Vector, args: &mut A);

    /// Accepts the update and computes the next stepsize
    ///
    /// Returns `stepsize_new`
    ///
    /// Note: thus function should compute and store the `relative_error`
    fn accept_update(&mut self, y0: &mut Vector, x0: f64, args: &mut A) -> f64;

    /// Rejects the update
    ///
    /// Returns the `relative_error`
    fn reject_update(&mut self) -> f64;

    /// Computes the dense output
    fn dense_output(&self, yout: &mut Vector, h: f64, x: f64, y: &Vector, xout: f64);
}
