use crate::StrError;
use crate::{NumSolver, OdeSystem, Workspace};
use russell_lab::{vec_add, vec_copy, Vector};
use russell_sparse::CooMatrix;
use std::marker::PhantomData;

pub(crate) struct EulerForward<'a, F, J, A>
where
    F: FnMut(&mut Vector, f64, &Vector, &mut A) -> Result<(), StrError>,
    J: FnMut(&mut CooMatrix, f64, &Vector, f64, &mut A) -> Result<(), StrError>,
{
    /// ODE system
    system: OdeSystem<'a, F, J, A>,

    /// Vector holding the function evaluation
    ///
    /// k := f(x, y)
    k: Vector,

    /// Auxiliary workspace (will contain y to be used in accept_update)
    w: Vector,

    /// Handle generic argument
    phantom: PhantomData<A>,
}

impl<'a, F, J, A> EulerForward<'a, F, J, A>
where
    F: FnMut(&mut Vector, f64, &Vector, &mut A) -> Result<(), StrError>,
    J: FnMut(&mut CooMatrix, f64, &Vector, f64, &mut A) -> Result<(), StrError>,
{
    /// Allocates a new instance
    pub fn new(system: OdeSystem<'a, F, J, A>) -> Self {
        let ndim = system.ndim;
        EulerForward {
            system,
            k: Vector::new(ndim),
            w: Vector::new(ndim),
            phantom: PhantomData,
        }
    }
}

impl<'a, F, J, A> NumSolver<A> for EulerForward<'a, F, J, A>
where
    F: FnMut(&mut Vector, f64, &Vector, &mut A) -> Result<(), StrError>,
    J: FnMut(&mut CooMatrix, f64, &Vector, f64, &mut A) -> Result<(), StrError>,
{
    /// Initializes the internal variables
    fn initialize(&mut self, _x: f64, _y: &Vector) {}

    /// Calculates the quantities required to update x and y
    fn step(&mut self, work: &mut Workspace, x: f64, y: &Vector, h: f64, args: &mut A) -> Result<(), StrError> {
        work.bench.n_function_eval += 1;
        (self.system.function)(&mut self.k, x, y, args)?; // k := f(x, y)
        vec_add(&mut self.w, 1.0, &y, h, &self.k).unwrap(); // w := y + h * f(x, y)
        Ok(())
    }

    /// Accepts the update and computes the next stepsize
    fn accept(
        &mut self,
        _work: &mut Workspace,
        y: &mut Vector,
        _x: f64,
        _h: f64,
        _args: &mut A,
    ) -> Result<(), StrError> {
        vec_copy(y, &self.w).unwrap();
        Ok(())
    }

    /// Rejects the update
    fn reject(&mut self, _work: &mut Workspace, _h: f64) {}

    /// Computes the dense output
    fn dense_output(&self, _y_out: &mut Vector, _h: f64, _x: f64, _x_out: f64) {}
}
