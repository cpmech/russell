use crate::constants::*;
use crate::ParamsRadau5;
use crate::StrError;
use crate::{Information, Method, NumSolver, ParamsERK, System, Workspace};
use russell_lab::{vec_copy, vec_update, Matrix, Vector};
use russell_sparse::{CooMatrix, Genie, LinSolver};

pub(crate) struct Radau5<'a, F, J, A>
where
    F: FnMut(&mut Vector, f64, &Vector, &mut A) -> Result<(), StrError>,
    J: FnMut(&mut CooMatrix, f64, &Vector, f64, &mut A) -> Result<(), StrError>,
{
    /// Holds the parameters
    params: ParamsRadau5,

    /// ODE system
    system: System<'a, F, J, A>,

    /// Linear solver
    solver: LinSolver<'a>,
}

impl<'a, F, J, A> Radau5<'a, F, J, A>
where
    F: FnMut(&mut Vector, f64, &Vector, &mut A) -> Result<(), StrError>,
    J: FnMut(&mut CooMatrix, f64, &Vector, f64, &mut A) -> Result<(), StrError>,
{
    /// Allocates a new instance
    pub fn new(method: Method, params: ParamsRadau5, system: System<'a, F, J, A>) -> Self {
        let ndim = system.ndim;
        let nnz = system.jac_nnz + ndim; // +ndim corresponds to the diagonal I matrix
        let symmetry = system.jac_symmetry;
        let one_based = if params.lin_sol == Genie::Mumps { true } else { false };
        Radau5 {
            params,
            system,
            solver: LinSolver::new(params.lin_sol).unwrap(),
        }
    }
}

impl<'a, F, J, A> NumSolver<A> for Radau5<'a, F, J, A>
where
    F: FnMut(&mut Vector, f64, &Vector, &mut A) -> Result<(), StrError>,
    J: FnMut(&mut CooMatrix, f64, &Vector, f64, &mut A) -> Result<(), StrError>,
{
    /// Initializes the internal variables
    fn initialize(&mut self, _x: f64, y: &Vector) {
        panic!("TODO");
    }

    /// Calculates the quantities required to update x and y
    fn step(&mut self, work: &mut Workspace, x: f64, y: &Vector, h: f64, args: &mut A) -> Result<(), StrError> {
        panic!("TODO");
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
        panic!("TODO");
        Ok(())
    }

    /// Rejects the update
    fn reject(&mut self, _work: &mut Workspace, _h: f64) {
        panic!("TODO");
    }

    /// Computes the dense output
    fn dense_output(&self, _y_out: &mut Vector, _h: f64, _x: f64, _x_out: f64) {}
}
