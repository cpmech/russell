use crate::StrError;
use crate::{OdeSolverTrait, System, Workspace};
use russell_lab::{vec_add, vec_copy, Vector};
use russell_sparse::CooMatrix;

pub(crate) struct EulerForward<'a, F, J, A>
where
    F: Send + FnMut(&mut Vector, f64, &Vector, &mut A) -> Result<(), StrError>,
    J: Send + FnMut(&mut CooMatrix, f64, &Vector, f64, &mut A) -> Result<(), StrError>,
{
    /// ODE system
    system: System<'a, F, J, A>,

    /// Vector holding the function evaluation
    ///
    /// k := f(x, y)
    k: Vector,

    /// Auxiliary workspace (will contain y to be used in accept_update)
    w: Vector,
}

impl<'a, F, J, A> EulerForward<'a, F, J, A>
where
    F: Send + FnMut(&mut Vector, f64, &Vector, &mut A) -> Result<(), StrError>,
    J: Send + FnMut(&mut CooMatrix, f64, &Vector, f64, &mut A) -> Result<(), StrError>,
{
    /// Allocates a new instance
    pub fn new(system: System<'a, F, J, A>) -> Self {
        let ndim = system.ndim;
        EulerForward {
            system,
            k: Vector::new(ndim),
            w: Vector::new(ndim),
        }
    }
}

impl<'a, F, J, A> OdeSolverTrait<A> for EulerForward<'a, F, J, A>
where
    F: Send + FnMut(&mut Vector, f64, &Vector, &mut A) -> Result<(), StrError>,
    J: Send + FnMut(&mut CooMatrix, f64, &Vector, f64, &mut A) -> Result<(), StrError>,
{
    /// Enables dense output
    fn enable_dense_output(&mut self) {}

    /// Initializes the internal variables
    fn initialize(&mut self, _work: &mut Workspace, _x: f64, _y: &Vector, _args: &mut A) -> Result<(), StrError> {
        Ok(())
    }

    /// Calculates the quantities required to update x and y
    fn step(&mut self, work: &mut Workspace, x: f64, y: &Vector, h: f64, args: &mut A) -> Result<(), StrError> {
        work.bench.n_function += 1;
        (self.system.function)(&mut self.k, x, y, args)?; // k := f(x, y)
        vec_add(&mut self.w, 1.0, &y, h, &self.k).unwrap(); // w := y + h * f(x, y)
        Ok(())
    }

    /// Updates x and y and computes the next stepsize
    fn accept(
        &mut self,
        _work: &mut Workspace,
        x: &mut f64,
        y: &mut Vector,
        h: f64,
        _args: &mut A,
    ) -> Result<(), StrError> {
        *x += h;
        vec_copy(y, &self.w).unwrap();
        Ok(())
    }

    /// Rejects the update
    fn reject(&mut self, _work: &mut Workspace, _h: f64) {}

    /// Computes the dense output with x-h ≤ x_out ≤ x
    fn dense_output(&self, _y_out: &mut Vector, _x_out: f64, _x: f64, _y: &Vector, _h: f64) -> Result<(), StrError> {
        Err("dense output is not available for the FwEuler method")
    }
}
