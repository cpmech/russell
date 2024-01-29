use crate::StrError;
use crate::{BenchInfo, NumSolver, OdeSystem};
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

    /// Holds benchmark information
    bench: BenchInfo,

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
            bench: BenchInfo::new(),
            phantom: PhantomData,
        }
    }
}

impl<'a, F, J, A> NumSolver<A> for EulerForward<'a, F, J, A>
where
    F: FnMut(&mut Vector, f64, &Vector, &mut A) -> Result<(), StrError>,
    J: FnMut(&mut CooMatrix, f64, &Vector, f64, &mut A) -> Result<(), StrError>,
{
    /// Returns an access to the benchmark structure
    fn bench(&mut self) -> &mut BenchInfo {
        &mut self.bench
    }

    /// Initializes the internal variables
    fn initialize(&mut self, _x: f64, _y: &Vector) {}

    /// Calculates the quantities required to update x and y
    ///
    /// Returns the (`relative_error`, `stiffness_ratio`)
    fn step(&mut self, x: f64, y: &Vector, h: f64, args: &mut A) -> Result<(f64, f64), StrError> {
        self.bench.n_function_eval += 1;
        (self.system.function)(&mut self.k, x, y, args)?; // k := f(x, y)
        vec_add(&mut self.w, 1.0, &y, h, &self.k).unwrap(); // w := y + h * f(x, y)
        Ok((0.0, 0.0))
    }

    /// Accepts the update and computes the next stepsize
    ///
    /// Returns `stepsize_new`
    fn accept(
        &mut self,
        y: &mut Vector,
        _x: f64,
        _h: f64,
        _relative_error: f64,
        _previous_relative_error: f64,
        _args: &mut A,
    ) -> Result<f64, StrError> {
        vec_copy(y, &self.w).unwrap();
        Ok(0.0)
    }

    /// Rejects the update
    ///
    /// Returns `stepsize_new`
    fn reject(&mut self, _h: f64, _relative_error: f64) -> f64 {
        0.0
    }

    /// Computes the dense output
    fn dense_output(&self, _y_out: &mut Vector, _h: f64, _x: f64, _x_out: f64) {}
}
