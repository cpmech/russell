use crate::StrError;
use crate::{OdeSolverTrait, OdeSys};
use russell_lab::{vec_add, vec_copy, vec_update, Vector};

pub struct EulerForward<A> {
    /// Dimension of the ODE system (len(y))
    ndim: usize,

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
            ndim,
            function,
            k0: Vector::new(ndim),
            w: Vector::new(ndim),
            n_function_eval: 0,
        }
    }
}

impl<A> OdeSolverTrait<A> for EulerForward<A> {
    fn step(&mut self, x0: f64, y0: &Vector, h: f64, first_step: bool, args: &mut A) -> (f64, f64) {
        (self.function)(&mut self.k0, x0, y0, args); // k0 := f(x0, y0)
        self.n_function_eval += 1;
        vec_add(&mut self.w, 1.0, &y0, h, &self.k0).unwrap(); // w := y0 + h * f(u0, y0)
        (0.0, 0.0)
    }

    fn accept(
        &mut self,
        y0: &mut Vector,
        x0: f64,
        h: f64,
        relative_error: f64,
        previous_relative_error: f64,
        args: &mut A,
    ) -> f64 {
        vec_copy(y0, &self.w).unwrap();
        0.0
    }

    fn reject(&mut self, h: f64, relative_error: f64) -> f64 {
        0.0
    }

    fn dense_output(&self, y_out: &mut Vector, h: f64, x: f64, y: &Vector, x_out: f64) {}
}
