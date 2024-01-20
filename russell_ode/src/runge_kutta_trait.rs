use crate::Information;

pub trait RungeKuttaTrait {
    /// Gathers information about the Runge-Kutta method
    fn information(&self) -> Information;

    fn initialize(&mut self);

    fn step(&mut self);

    /// Accepts the update
    ///
    /// Returns `(stepsize_new, relative_error)`
    fn accept_update(&mut self) -> (f64, f64);

    /// Rejects the update
    ///
    /// Returns the `relative_error`
    fn reject_update(&mut self) -> f64;

    fn dense_output(&self);
}
