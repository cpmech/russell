use crate::NumSolver;
use crate::StrError;
use russell_lab::{vec_add, vec_copy, Vector};

pub(crate) struct EulerForward<F>
where
    F: FnMut(&mut Vector, f64, &Vector) -> Result<(), StrError>,
{
    /// ODE system
    system: F,

    /// Vector holding the function evaluation
    ///
    /// k := f(x, y)
    k: Vector,

    /// Auxiliary workspace (will contain y to be used in accept_update)
    w: Vector,

    /// number of calls to function
    n_function_eval: usize,
}

impl<F> EulerForward<F>
where
    F: FnMut(&mut Vector, f64, &Vector) -> Result<(), StrError>,
{
    /// Allocates a new instance
    pub fn new(ndim: usize, system: F) -> Self {
        EulerForward {
            system,
            k: Vector::new(ndim),
            w: Vector::new(ndim),
            n_function_eval: 0,
        }
    }
}

impl<F> NumSolver for EulerForward<F>
where
    F: FnMut(&mut Vector, f64, &Vector) -> Result<(), StrError>,
{
    /// Initializes the internal variables
    fn initialize(&mut self, _x: f64, _y: &Vector) {
        self.n_function_eval = 0;
    }

    /// Calculates the quantities required to update x and y
    ///
    /// Returns the (`relative_error`, `stiffness_ratio`)
    fn step(&mut self, x: f64, y: &Vector, h: f64) -> Result<(f64, f64), StrError> {
        self.n_function_eval += 1;
        (self.system)(&mut self.k, x, y)?; // k := f(x, y)
        vec_add(&mut self.w, 1.0, &y, h, &self.k).unwrap(); // w := y + h * f(x, y)
        Ok((0.0, 0.0))
    }

    /// Accepts the update and computes the next stepsize
    ///
    /// Returns `stepsize_new`
    fn accept(&mut self, y: &mut Vector, _: f64, _: f64, _: f64, _: f64) -> Result<f64, StrError> {
        vec_copy(y, &self.w).unwrap();
        Ok(0.0)
    }

    /// Rejects the update
    ///
    /// Returns `stepsize_new`
    fn reject(&mut self, _: f64, _: f64) -> f64 {
        0.0
    }

    /// Computes the dense output
    fn dense_output(&self, _: &mut Vector, _: f64, _: f64, _: f64) {}
}
