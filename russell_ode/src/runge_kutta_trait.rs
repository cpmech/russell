use crate::Information;

pub trait RungeKuttaTrait {
    /// Gathers information about the Runge-Kutta method
    fn information(&self) -> Information;

    /// Initializes the solver
    fn initialize(&mut self);

    /// Performs  the next step
    fn next_step(&mut self);

    /// Accepts the update and computes the next stepsize
    ///
    /// Returns `(stepsize_new, relative_error)`
    fn accept_update(&mut self) -> (f64, f64);

    /// Rejects the update
    ///
    /// Returns the `relative_error`
    fn reject_update(&mut self) -> f64;

    /// Computes the dense output
    fn dense_output(&self);
}
