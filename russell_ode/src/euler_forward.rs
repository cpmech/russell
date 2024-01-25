use crate::StrError;
use crate::{OdeSolverTrait, OdeSys};
use russell_lab::{vec_add, vec_copy, Vector};

pub struct EulerForward<A> {
    /// Function defining the ODE problem
    ///
    /// dy/dx = f(x, y)
    function: OdeSys<A>,

    /// Vector holding the function evaluation
    ///
    /// k0 := f(x0, y0)
    k0: Vector,

    /// Auxiliary workspace (will contain y0 to be used in accept_update)
    w: Vector,

    /// number of calls to function
    n_function_eval: usize,
}

impl<A> EulerForward<A> {
    pub fn new(ndim: usize, function: OdeSys<A>) -> Self {
        EulerForward {
            function,
            k0: Vector::new(ndim),
            w: Vector::new(ndim),
            n_function_eval: 0,
        }
    }
}

impl<A> OdeSolverTrait<A> for EulerForward<A> {
    fn step(&mut self, x0: f64, y0: &Vector, h: f64, _: bool, args: &mut A) -> Result<(f64, f64), StrError> {
        (self.function)(&mut self.k0, x0, y0, args)?; // k0 := f(x0, y0)
        self.n_function_eval += 1;
        vec_add(&mut self.w, 1.0, &y0, h, &self.k0).unwrap(); // w := y0 + h * f(u0, y0)
        Ok((0.0, 0.0))
    }

    fn accept(&mut self, y0: &mut Vector, _: f64, _: f64, _: f64, _: f64, _: &mut A) -> Result<f64, StrError> {
        vec_copy(y0, &self.w).unwrap();
        Ok(0.0)
    }

    fn reject(&mut self, _: f64, _: f64) -> f64 {
        0.0
    }

    fn dense_output(&self, _: &mut Vector, _: f64, _: f64, _: f64) {}
}
